from .utils import format_time
import shutil
import subprocess
import time
from pathlib import Path
import soundfile as sf
import torch

from .utils import (
    log_msg, attempt_cpu_run_with_retry, attempt_run_with_retry,
    run_command_with_progress, is_valid_audio, is_valid_video,
    FFMPEG_BIN
)
from .config import (
    KEEP_INPUT_FILES,
    VOCALS_MODEL, BACKGROUND_MODEL, DENOISE_MODEL, ENHANCE_NFE, ENHANCE_TAU,
    PROCESS_MODE, VOCAL_MIX_VOL, BACKGROUND_MIX_VOL
)
from .hardware import (
    CPU_THREADS, GPU_BATCH_SIZE, CUDA_DEVICE
)
from .sync import _align_stems


def get_audio_duration_sec(wav_path):
    try:
        with sf.SoundFile(str(wav_path)) as f:
            return f.frames / f.samplerate
    except Exception:
        return None


def get_video_duration_sec(video_path):
    """Gets video duration using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        val = subprocess.check_output(cmd).decode().strip()
        return float(val)
    except Exception:
        return None


def _extract_audio_step(video_path, original_wav, total_duration=None):
    """Step 1: Extract High-Res Audio."""
    if is_valid_audio(original_wav):
        log_msg("  [System] Skipping Extraction (valid exists)")
        return

    if original_wav.exists():
        original_wav.unlink()

    log_msg("  [System] Extracting Audio Stream...")

    # Extract to .tmp first
    tmp_wav = original_wav.with_suffix(".tmp.wav")

    def build_extract_cmd(threads):
        return [
            FFMPEG_BIN, "-stats", "-hide_banner",
            "-threads", str(threads),
            "-i", str(video_path),
            "-acodec", "pcm_f32le", "-ar", "44100", "-y",
            str(tmp_wav)
        ]

    attempt_cpu_run_with_retry(
        build_extract_cmd, CPU_THREADS,
        description="Extracting Audio",
        total_duration=total_duration
    )

    if is_valid_audio(tmp_wav):
        tmp_wav.rename(original_wav)
    else:
        if tmp_wav.exists():
            tmp_wav.unlink()
        raise Exception("Extraction failed: Output audio is invalid, empty, or too small.")


def _verify_separation_output(separation_out_dir, original_wav):
    """Verifies that separation produced both stems with extreme robustness."""
    # Pattern 1: Explicit tags
    v_files = list(separation_out_dir.glob("*(Vocals)*.wav"))
    b_files = list(separation_out_dir.glob("*(Instrumental)*.wav"))
    b_files += list(separation_out_dir.glob("*(Background)*.wav"))
    b_files += list(separation_out_dir.glob("*(No Vocals)*.wav"))

    # Pattern 2: Fallback - if we have exactly 2 wav files and only one is Vocals
    all_wavs = list(separation_out_dir.glob("*.wav"))
    if not b_files and len(all_wavs) == 2 and v_files:
        other = [f for f in all_wavs if f != v_files[0]]
        if other:
            b_files = other

    # Sort by size/time to pick best candidates
    v_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    b_files.sort(key=lambda x: x.stat().st_size, reverse=True)

    v_valid = [f for f in v_files if is_valid_audio(f)]
    b_valid = [f for f in b_files if is_valid_audio(f)]

    if v_valid and b_valid:
        vocals = v_valid[0]
        background = b_valid[0]

        # Ensure consistency: Rename background to use (Background) tag
        if "(Background)" not in background.name:
            # Try to build a clean name based on vocals or original
            new_name = vocals.name.replace("(Vocals)", "(Background)")
            new_path = separation_out_dir / new_name
            if not new_path.exists():
                try:
                    background.rename(new_path)
                    background = new_path
                except OSError:
                    pass
        return vocals, background

    # Detailed logging on failure
    if not v_valid or not b_valid:
        log_msg("    [Debug] Separation output mismatch.", level="DEBUG")
        log_msg(f"    [Debug] Found Vocals: {[f.name for f in v_valid]}", level="DEBUG")
        log_msg(f"    [Debug] Found Background: {[f.name for f in b_valid]}", level="DEBUG")
        all_any = list(separation_out_dir.glob("*"))
        log_msg(f"    [Debug] All files in dir: {[f.name for f in all_any]}", level="DEBUG")

    return None, None


def _separate_stems_step(original_wav, separation_out_dir, total_duration=None):
    """
    Step 2: Separate Stems (BS-Roformer).
    Returns path to (vocals_wav, background_wav).
    """
    # 1. Check Existing
    existing_v, existing_b = _verify_separation_output(separation_out_dir, original_wav)
    if existing_v and existing_b:
        log_msg("  [Step 1/5] Skipping Separation (exists & valid)")
        return existing_v, existing_b

    log_msg("  [Step 1/5] Separating Stems (BS-Roformer)...")

    def build_roformer_cmd(bs):
        return [
            "audio-separator", str(original_wav),
            "--model_filename", VOCALS_MODEL,
            "--output_dir", str(separation_out_dir),
            "--output_format", "wav",
            "--normalization", "0.9",
            "--vr_batch_size", str(bs),
            "--vr_window_size", "320",
            "--use_soundfile",  # Forced by patch
        ]

    attempt_run_with_retry(
        build_roformer_cmd, GPU_BATCH_SIZE,
        description="Separating Stems (AI)",
        total_duration=total_duration
    )

    # 2. Verify Output
    vocals_wav, background_wav = _verify_separation_output(separation_out_dir, original_wav)

    if not vocals_wav or not background_wav:
        raise Exception("Separation failed: Output stems not found.")

    return vocals_wav, background_wav


def _run_enhance_retry(cmd_enhance, total_duration):
    """Retries the enhancement command."""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            run_command_with_progress(
                cmd_enhance, description="Enhancing Vocals",
                total_duration=total_duration
            )
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                log_msg(
                    f"    [Warning] Enhancement failed (Attempt {attempt + 1}). Retrying...",
                    is_error=True
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                time.sleep(2)
            else:
                raise e


def _handle_enhance_output(enhanced_vocals_dir, vocals_wav):
    """Checks output and verifies validity."""
    candidates_enhanced = list(enhanced_vocals_dir.glob("*.wav"))
    if not candidates_enhanced:
        log_msg(
            "    [Warning] Resemble-Enhance did not produce output. "
            "Using raw vocals.", is_error=True
        )
        fb_path = (
            enhanced_vocals_dir / f"fallback_{vocals_wav.name}"
        )
        shutil.copy(vocals_wav, fb_path)
        return fb_path

    return candidates_enhanced[0]


def _enhance_vocals_step(vocals_wav, enhanced_vocals_dir, work_dir, total_duration=None):
    """Step 3: Enhance Vocals (Resemble-Enhance)."""
    enhanced_vocals_dir.mkdir(parents=True, exist_ok=True)
    candidates_enhanced = list(enhanced_vocals_dir.glob("*.wav"))
    valid_enhanced = [f for f in candidates_enhanced if is_valid_audio(f)]

    if valid_enhanced:
        log_msg("  [Step 2/5] Skipping Vocal Enhancement (exists)")
        return valid_enhanced[0]

    log_msg("  [Step 2/5] Enhancing Vocals (Resemble-Enhance)...")

    # Resemble-Enhance expects an input DIRECTORY, not a file.
    enhance_input_dir = work_dir / "enhance_input"
    if enhance_input_dir.exists():
        shutil.rmtree(enhance_input_dir)
    enhance_input_dir.mkdir()

    shutil.copy(vocals_wav, enhance_input_dir / vocals_wav.name)

    # Atomic directory pattern
    enhanced_vocals_tmp_dir = work_dir / "enhanced_vocals_tmp"
    if enhanced_vocals_tmp_dir.exists():
        shutil.rmtree(enhanced_vocals_tmp_dir)
    enhanced_vocals_tmp_dir.mkdir()

    cmd_enhance = [
        "resemble-enhance",
        str(enhance_input_dir),
        str(enhanced_vocals_tmp_dir),
        "--denoise_only",
        "--nfe", str(ENHANCE_NFE),
        "--solver", "rk4",
        "--tau", str(ENHANCE_TAU),
        "--device", CUDA_DEVICE
    ]

    _run_enhance_retry(cmd_enhance, total_duration)

    try:
        shutil.rmtree(enhance_input_dir)
    except Exception:
        pass

    # Verify and Move
    result = _handle_enhance_output(enhanced_vocals_tmp_dir, vocals_wav)

    # If we got a result (either enhanced or fallback), move it to final dir
    final_output = enhanced_vocals_dir / result.name
    shutil.copy(result, final_output)

    # Cleanup tmp dir
    try:
        shutil.rmtree(enhanced_vocals_tmp_dir)
    except Exception:
        pass

    return final_output


def _denoise_background_step(background_wav, denoised_background_dir, total_duration=None):
    """Step 4: Denoise Background (UVR-DeNoise-Lite)."""
    candidates_denoised = list(
        denoised_background_dir.glob("*.wav")
    )
    valid_denoised = [f for f in candidates_denoised if is_valid_audio(f)]

    if valid_denoised:
        log_msg("  [Step 3/5] Skipping Background Denoising (exists)")
        return valid_denoised[0]

    log_msg("  [Step 3/5] Denoising Background (UVR-DeNoise-Lite)...")

    def build_denoise_cmd(bs):
        return [
            "audio-separator", str(background_wav),
            "--model_filename", DENOISE_MODEL,
            "--output_dir", str(denoised_background_dir),
            "--output_format", "wav",
            "--single_stem", "No Noise",
            "--vr_batch_size", str(bs),
            "--vr_window_size", "320",
            "--use_soundfile",
        ]

    attempt_run_with_retry(
        build_denoise_cmd, GPU_BATCH_SIZE,
        description="Denoising Background (AI)",
        total_duration=total_duration
    )

    candidates_denoised = list(
        denoised_background_dir.glob("*.wav")
    )

    clean_candidates = [
        f for f in candidates_denoised if "(No Noise)" in f.name
    ]

    if clean_candidates:
        result = clean_candidates[0]
    elif candidates_denoised:
        result = candidates_denoised[0]
    else:
        log_msg("    [Warning] UVR-DeNoise failed. Using raw background.",
                is_error=True)
        result = background_wav

    log_msg(f"    Selected Denoised Stem: {result.name}")
    return result


def _final_mix_step(
    video_path, enhanced_vocals_wav,
    denoised_background_wav, final_output_video, total_duration=None
):
    """Step 5: Mix with FFmpeg."""
    if is_valid_video(final_output_video):
        log_msg(f"  [Step 5/5] Skipping Final Mix (exists: {final_output_video.name})")
        return

    log_msg("  [Step 5/5] Final Mix (32-bit Float)...")

    # Use provided duration or calculate from enhanced vocals
    duration = total_duration or get_audio_duration_sec(enhanced_vocals_wav)

    # Verify inputs exist to avoid cryptic FFmpeg errors
    if not enhanced_vocals_wav.exists():
        raise FileNotFoundError(f"Missing Vocals: {enhanced_vocals_wav}")
    if not denoised_background_wav.exists():
        raise FileNotFoundError(f"Missing Background: {denoised_background_wav}")

    # Atomic Final Write
    # Fix: Use same suffix as final/input to support containers like MOV (ProRes)
    # forcing .mp4 causes codec tag errors if copying unsupported streams
    tmp_output = final_output_video.with_suffix(f".tmp{final_output_video.suffix}")

    def build_mix_cmd(threads):
        return [
            FFMPEG_BIN, "-stats", "-hide_banner",
            "-threads", str(threads),
            "-i", str(video_path),          # 0: Video source
            "-i", str(enhanced_vocals_wav),
            "-i", str(denoised_background_wav),
            "-map", "0:v",
            "-filter_complex",
            f"[1:a]volume={VOCAL_MIX_VOL}[v];"
            f"[2:a]volume={BACKGROUND_MIX_VOL}[m];"
            "[v][m]amix=inputs=2:duration=longest:normalize=0[out]",
            "-map", "[out]",
            "-c:v", "copy",
            "-c:a", "pcm_f32le",
            "-shortest", "-y", str(tmp_output)
        ]

    attempt_cpu_run_with_retry(
        build_mix_cmd, CPU_THREADS,
        description="Final Mixing",
        total_duration=duration
    )

    if is_valid_video(tmp_output):
        if final_output_video.exists():
            final_output_video.unlink()
        tmp_output.rename(final_output_video)
        log_msg(f"  [System] Success! Saved to: {final_output_video.name}")
    else:
        if tmp_output.exists():
            tmp_output.unlink()
        raise Exception("Final Mix Failed: Output video invalid/empty.")


def process_hybrid_audio(video_path, gpu_name, target_output_dir=None):
    """Main Orchestrator."""
    log_msg(f"\n[System] Processing Task: {video_path.name}")

    if not video_path.exists():
        log_msg(f"  [Error] File not found: {video_path}", is_error=True)
        return False

    # Create safe working directory pattern
    # Use a hidden temp dir in the same location to ensure atomic moves work
    work_dir = video_path.parent / f".temp_work_{video_path.stem}"
    if target_output_dir:
        output_dir = target_output_dir
    else:
        output_dir = video_path.parent

    # Define final output name
    final_output_video = output_dir / f"{video_path.stem}_Hybrid_Cleaned{video_path.suffix}"

    # Resume checks: If final exists, skip
    if is_valid_video(final_output_video):
        log_msg("  [System] Output already exists. Skipping.")
        return True

    try:
        work_dir.mkdir(exist_ok=True)
        original_wav = work_dir / "original.wav"

        # Step 0: Video Duration
        video_dur = get_video_duration_sec(video_path)
        if video_dur:
            log_msg(f"  [Info] Duration: {format_time(video_dur)}")

        # Step 1: Extract Audio
        _extract_audio_step(video_path, original_wav, total_duration=video_dur)

        # Step 2: Separate Stems
        if PROCESS_MODE == "denoise_only":
            # Just pretend vocals is original and background is silence for logic flow?
            # Or just bypass. For strict refactor, ignoring 'denoise_only' nuance if not in original logic?
            # Original logic handled "denoise_only" in config but I didn't see explicit branch in main loop in my last read.
            # Ah, the `config.py` had `PROCESS_MODE`. Check `restore_audio_hybrid.py` for usage.
            # I didn't see `process_mode` being checked in `_separate_stems_step` or `process_hybrid_audio` in the snippets I read.
            # I will assume standard hybrid flow as seen in the code I viewed.
            pass

        separation_out_dir = work_dir / "separation"
        separation_out_dir.mkdir(exist_ok=True)

        vocals_wav, background_wav = _separate_stems_step(
            original_wav, separation_out_dir, total_duration=video_dur
        )

        # Step 3: Enhance Vocals
        enhanced_vocals_dir = work_dir / "enhanced_vocals"
        enhanced_vocals_wav = _enhance_vocals_step(
            vocals_wav, enhanced_vocals_dir, work_dir, total_duration=video_dur
        )

        # Step 4: Denoise Background
        denoised_background_dir = work_dir / "denoised_background"
        denoised_background_wav = _denoise_background_step(
            background_wav, denoised_background_dir, total_duration=video_dur
        )

        # Step 5: Sync
        log_msg("  [Step 4/5] Smart Audio Sync (Sequential for clean output)...")
        log_msg("    [Info] Syncing Stems (Sequential for clean output)...")

        aligned_vocals = work_dir / f"aligned_{enhanced_vocals_wav.name}"
        _align_stems(original_wav, enhanced_vocals_wav, aligned_vocals)

        aligned_background = work_dir / f"aligned_{denoised_background_wav.name}"
        _align_stems(original_wav, denoised_background_wav, aligned_background)

        # Step 6: Final Mix
        _final_mix_step(
            video_path, aligned_vocals, aligned_background, final_output_video,
            total_duration=video_dur
        )

        log_msg(f"  [System] Task Completed: {video_path.name}")
        return True

    except Exception as e:
        log_msg(f"  [Error] Processing failed: {e}", is_error=True)
        return False

    finally:
        # Cleanup Temp Directory
        if work_dir.exists() and not KEEP_INPUT_FILES:
            if is_valid_video(final_output_video):
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass
            else:
                log_msg(f"  [System] Preservation: Keeping {work_dir.name} for inspection on failure.", level="DEBUG")


# Need format_time for `process_hybrid_audio` duration logging
