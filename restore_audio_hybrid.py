import os
import subprocess
import shutil
import sys
import platform
import datetime
from pathlib import Path
import re

from typing import Set
import signal
import atexit
import time
import concurrent.futures

import soundfile as sf  # type: ignore
import torch
import yaml
import numpy as np
import scipy.signal

# === AUTO-CONFIGURE PATH ===
project_dir = Path(__file__).parent.resolve()
venv_scripts = project_dir / "venv" / "Scripts"

FFMPEG_BIN = "ffmpeg"
if (venv_scripts / "ffmpeg.exe").exists():
    FFMPEG_BIN = str(venv_scripts / "ffmpeg.exe")
    current_path = os.environ.get("PATH", "")
    if str(venv_scripts) not in current_path:
        os.environ["PATH"] = str(venv_scripts) + os.pathsep + current_path
elif venv_scripts.exists():  # pragma: no cover
    current_path = os.environ.get("PATH", "")
    if str(venv_scripts) not in current_path:
        os.environ["PATH"] = str(venv_scripts) + os.pathsep + current_path
else:
    pass  # Will log after log_msg is defined

_venv_scripts_missing = not venv_scripts.exists()

# === SUBPROCESS MANAGEMENT ===
_active_processes: Set[subprocess.Popen] = set()


def cleanup_subprocesses():
    """Terminates all registered active subprocesses."""
    if not _active_processes:
        return

    log_msg(f"\n[System] Cleaning up {_len_active()} processes...", level="DEBUG")
    # Work on a copy to avoid 'Set changed size during iteration'
    for p in list(_active_processes):
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:  # pragma: no cover
            pass

    # Give them a moment to terminate gracefully
    time.sleep(0.5)

    for p in list(_active_processes):
        try:
            if p.poll() is None:
                p.kill()
        except Exception:  # pragma: no cover
            pass
    _active_processes.clear()


def _len_active():
    return len(_active_processes)


def signal_handler(sig, frame):  # pragma: no cover
    """Handles termination signals."""
    log_msg("\n[System] Termination signal received. Stopping...", is_error=True)
    cleanup_subprocesses()
    sys.exit(1)


# Register handlers
atexit.register(cleanup_subprocesses)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if platform.system() == "Windows":  # pragma: no cover
    sig_break = getattr(signal, "SIGBREAK", None)
    if sig_break is not None:
        signal.signal(sig_break, signal_handler)
# ============================


def load_config():
    config_path = Path("config.yaml")
    defaults = {
        "vocal_mix_volume": 1.0,
        "music_mix_volume": 1.0,
        "extensions": ['.mp4', '.mkv', '.avi', '.mov'],
        "vocals_model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "instrumental_model": "UVR-MDX-NET-Inst_HQ_3.onnx",
        "denoise_model": "UVR-DeNoise-Lite.pth",
        "enhance_nfe": 128,
        "enhance_tau": 0.5
    }
    loaded_from = "Defaults"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    defaults.update(user_config)
                    loaded_from = "config.yaml"
        except Exception as e:
            print(f"[Warning] Failed to load config.yaml: {e}")

    return defaults, loaded_from


CONFIG, CONFIG_SOURCE = load_config()

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp_work")
LOG_FILE = Path("session_log.txt")

EXTS = set(CONFIG["extensions"])
KEEP_INPUT_FILES = os.environ.get("AI_RESTORE_TEST_MODE") == "1"

# Audio mix levels
VOCAL_MIX_VOL = CONFIG["vocal_mix_volume"]
MUSIC_MIX_VOL = CONFIG["music_mix_volume"]

# AI Configs
VOCALS_MODEL = CONFIG["vocals_model"]
INSTRUMENTAL_MODEL = CONFIG["instrumental_model"]
DENOISE_MODEL = CONFIG["denoise_model"]
ENHANCE_NFE = str(CONFIG["enhance_nfe"])
ENHANCE_TAU = str(CONFIG["enhance_tau"])


def get_optimal_settings():
    """Auto-detect hardware and return optimal settings."""
    settings = {
        "cpu_threads": os.cpu_count() or 16,
        "gpu_batch_size": 1,
        "cuda_device": "cuda:0",
        "gpu_vram_gb": 0,
        "profile_name": "Low (Entry Config)"
    }

    # Detect GPU VRAM and calculate optimal batch size
    if torch.cuda.is_available():
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024 ** 3)
            settings["gpu_vram_gb"] = vram_gb

            if vram_gb >= 24:
                settings["gpu_batch_size"] = 32
                settings["profile_name"] = "EXTREME (RTX 5090 / A6000)"
            elif vram_gb >= 22:
                settings["gpu_batch_size"] = 16
                settings["profile_name"] = "ULTRA (RTX 3090/4090)"
            elif vram_gb >= 15:
                settings["gpu_batch_size"] = 8
                settings["profile_name"] = "HIGH (RTX 4080/5080)"
            elif vram_gb >= 10:
                settings["gpu_batch_size"] = 4
                settings["profile_name"] = "MID (RTX 3080/4070)"
            else:
                settings["gpu_batch_size"] = 1
                settings["profile_name"] = "LOW (Entry Config)"

        except Exception:  # pragma: no cover
            pass  # Keep defaults

    return settings


# Auto-configure on import
_hw_settings = get_optimal_settings()


CPU_THREADS = _hw_settings["cpu_threads"]
GPU_BATCH_SIZE = _hw_settings["gpu_batch_size"]
CUDA_DEVICE = _hw_settings["cuda_device"]
GPU_VRAM_GB = _hw_settings["gpu_vram_gb"]
PROFILE_NAME = _hw_settings["profile_name"]


def format_time(seconds):
    """Formats seconds as HH:MM:SS.mm"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def draw_progress_bar(
    percent, label="", width=30, elapsed_sec=None, media_sec=None
):
    """
    Draws a modern visual progress bar with optional time and speed info.
    Format: [██...░░] 53.30% | 00:17:02.28 | 2.57x | ETA 00:15:30.00 | Rendering...
    """
    percent = max(0, min(100, percent))
    filled_length = int(width * percent // 100)
    bar = "█" * filled_length + "░" * (width - filled_length)

    # Build info string
    info_parts = [f"{percent:6.2f}%"]

    if elapsed_sec is not None:
        info_parts.append(format_time(elapsed_sec))

        # Calculate speed if we have both elapsed and media time
        if media_sec is not None and elapsed_sec > 0:
            speed = media_sec / elapsed_sec
            info_parts.append(f"{speed:.2f}x")

        # Calculate ETA (Estimated Time Remaining)
        if percent > 0 and percent < 100:
            eta_sec = (elapsed_sec / percent) * (100 - percent)
            info_parts.append(f"ETA {format_time(eta_sec)}")

    info_parts.append(label)
    info_str = " | ".join(info_parts)

    # \r overwrites the line, \033[K clears to the end of the line
    sys.stdout.write(f"\r\033[K    [{bar}] {info_str}")
    sys.stdout.flush()

    if percent >= 100:
        sys.stdout.write("\n")


def log_msg(message, is_error=False, console=True, level="INFO"):
    """
    Logs messages to console and log file.

    Args:
        message: The message to log.
        is_error: If True, marks as ERROR level (overrides 'level').
        console: If True, prints to console (unless level is DEBUG).
        level: Log level - 'INFO', 'DEBUG', or 'ERROR'. DEBUG never prints to console.
    """
    # Determine effective log level
    if is_error:
        effective_level = "ERROR"
    else:
        effective_level = level.upper()

    # DEBUG logs never print to console
    should_print = console and effective_level != "DEBUG"

    if should_print:
        print(message)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_msg = message.strip()

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{effective_level:5}] {clean_msg}\n")
    except Exception:
        pass


# Deferred debug log from module init
if _venv_scripts_missing:
    log_msg(f"Venv Scripts not found at: {project_dir / 'venv' / 'Scripts'}", level="DEBUG")


def is_valid_audio(file_path):
    """
    Checks if a file exists, has content, and has a valid audio header.
    Returns False if corrupted or empty.
    """
    path = Path(file_path)
    if not path.exists():
        return False

    # Less than 1KB is likely not a valid processed audio file
    if path.stat().st_size < 1024:
        return False

    try:
        # Quick header check using SoundFile
        with sf.SoundFile(str(path)) as f:
            if f.frames > 0:
                return True
    except Exception:
        pass

    return False


def get_nvidia_paths():
    """Returns a list of paths containing CUDNN/CUBLAS DLLs."""
    nvidia_paths = []

    # Try torch.lib first
    try:
        import torch
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists():
            nvidia_paths.append(str(torch_lib))
    except ImportError:
        pass

    # Try nvidia.* packages
    try:
        import nvidia.cudnn  # type: ignore
        import nvidia.cublas  # type: ignore
        for lib in [nvidia.cudnn, nvidia.cublas]:
            if hasattr(lib, '__path__') and lib.__path__:
                path = lib.__path__[0]
            else:
                path = os.path.dirname(lib.__file__)
            nvidia_paths.append(os.path.join(path, "bin"))
            nvidia_paths.append(os.path.join(path, "lib"))
    except ImportError:
        pass

    return nvidia_paths


def get_cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
            processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            return processor_name.strip()
        except Exception:  # pragma: no cover
            pass
    return platform.processor() or "Unknown CPU"


def get_gpu_name():
    try:
        output = subprocess.check_output("nvidia-smi -L", shell=True).decode()
        # Format: GPU 0: NVIDIA GeForce RTX 5090 (UUID: ...)
        if "NVIDIA" in output:
            return output.split(':')[1].split('(')[0].strip()
    except Exception:  # pragma: no cover
        pass
    return "Generic / Not Detected"


def check_dependencies():
    missing = []

    try:
        subprocess.run(
            [FFMPEG_BIN, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        missing.append("FFmpeg")

    try:
        subprocess.run(
            ["audio-separator", "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        missing.append("Audio-Separator")

    if missing:
        log_msg(f"CRITICAL: Missing: {', '.join(missing)}", is_error=True)
        log_msg(f"Search Path: {os.environ['PATH']}", is_error=True)
        return False
    return True


def get_audio_duration_sec(file_path):
    """Returns duration of audio file in seconds."""
    try:
        with sf.SoundFile(str(file_path)) as f:
            return float(f.frames) / float(f.samplerate)
    except Exception:  # pragma: no cover
        return 0.0


def parse_ffmpeg_time(line):
    """Extracts time=HH:MM:SS.mm from FFmpeg output."""
    match = re.search(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})", line)
    if match:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 100.0
    return None


def run_command_with_progress(
    cmd, env=None, description="Running...", total_duration=None
):
    """
    Runs a subprocess.
    If total_duration is set, parses stderr for FFmpeg progress.
    Otherwise, just pipes output (for TQDM/Audio-Separator).
    """
    if total_duration is None:
        # Standard passthrough for TQDM-based tools (Audio-Separator)
        print(f"      {description}")
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        _active_processes.add(process)
        try:
            process.wait()
        finally:
            _active_processes.discard(process)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        return

    # FFmpeg Progress Parsing with enhanced display
    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace"  # Handle potential encoding issues
    )
    _active_processes.add(process)

    duration = max(0.1, total_duration)

    try:
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break

            if line:
                current_time = parse_ffmpeg_time(line)
                if current_time is not None:
                    percent = (current_time / duration) * 100
                    elapsed = time.time() - start_time
                    draw_progress_bar(
                        percent, description,
                        elapsed_sec=elapsed, media_sec=current_time
                    )
    finally:
        _active_processes.discard(process)

    process.wait()

    # Ensure 100% on success
    if process.returncode == 0:
        elapsed = time.time() - start_time
        draw_progress_bar(
            100.0, description,
            elapsed_sec=elapsed, media_sec=duration
        )
    else:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def attempt_run_with_retry(
    command_builder_func, initial_batch_size, description="Running..."
):
    """
    Retries a command with reduced GPU batch sizes on OOM.
    command_builder_func: Accepts 'batch_size' (int), returns [cmd, args...].
    """
    current_batch_size = initial_batch_size

    while True:
        try:
            cmd = command_builder_func(current_batch_size)
            print(f"      {description} (Batch Size: {current_batch_size})")

            process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            _active_processes.add(process)
            try:
                process.wait()
            finally:
                _active_processes.discard(process)

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            return True  # Success

        except subprocess.CalledProcessError as e:
            if current_batch_size > 1:
                log_msg(
                    "    [Warning] GPU failed. Retrying with reduced batch...",
                    is_error=True
                )
                current_batch_size = max(1, current_batch_size // 2)
                if torch.cuda.is_available():  # pragma: no cover
                    torch.cuda.empty_cache()
                import gc  # pragma: no cover
                gc.collect()
            else:
                raise e


def attempt_cpu_run_with_retry(
    command_builder_func, initial_threads, description="Running...",
    total_duration=None
):
    """
    Retries a CPU-bound command with reduced threads on RAM OOM.
    command_builder_func: Accepts 'threads' (int), returns [cmd, args...].
    """
    current_threads = initial_threads

    while True:
        try:
            cmd = command_builder_func(current_threads)
            # If we don't have duration, use standard print
            if total_duration is None:
                print(f"      {description} (Threads: {current_threads})")

            run_command_with_progress(
                cmd,
                description=description,
                total_duration=total_duration
            )

            return True  # Success

        except subprocess.CalledProcessError as e:
            if current_threads > 1:
                log_msg(
                    "    [Warning] CPU failed. Retrying with fewer threads...",
                    is_error=True
                )
                current_threads = max(1, current_threads // 2)

                # Cleanup RAM
                import gc  # pragma: no cover
                gc.collect()
            else:
                raise e


def _extract_audio_step(video_path, original_wav):
    """Step 1: Extract High-Res Audio."""
    if is_valid_audio(original_wav):
        log_msg("  [System] Skipping Extraction (exists)")
    else:
        log_msg("  [System] Extracting Audio Stream...")

        def build_extract_cmd(threads):
            return [
                FFMPEG_BIN, "-stats", "-hide_banner",
                "-threads", str(threads),
                "-i", str(video_path),
                "-acodec", "pcm_f32le", "-ar", "44100", "-y",
                str(original_wav)
            ]

        attempt_cpu_run_with_retry(
            build_extract_cmd, CPU_THREADS,
            description="Extracting Audio"
        )


def _separate_stems_step(original_wav, separation_out_dir):
    """
    Step 2: Separate Stems (BS-Roformer).
    Returns path to (vocals_wav, background_wav).
    Using BS-Roformer for BOTH stems ensures 'Background' is simply
    'Original - Vocals', preserving ambient sounds (birds, etc.) that
    other models might aggressively filter out.
    """
    v_glob = f"{original_wav.stem}_(Vocals)_model_bs_roformer*.wav"
    i_glob = f"{original_wav.stem}_(Instrumental)_model_bs_roformer*.wav"

    existing_vocals = list(separation_out_dir.glob(v_glob))
    existing_back = list(separation_out_dir.glob(i_glob))

    v_valid = [f for f in existing_vocals if is_valid_audio(f)]
    b_valid = [f for f in existing_back if is_valid_audio(f)]

    if v_valid and b_valid:
        log_msg("  [Step 1/5] Skipping Separation (exists)")
        return v_valid[0], b_valid[0]

    log_msg("  [Step 1/5] Separating Stems (BS-Roformer)...")

    def build_roformer_cmd(bs):
        return [
            "audio-separator", str(original_wav),
            "--model_filename", VOCALS_MODEL,
            "--output_dir", str(separation_out_dir),
            "--output_format", "wav",
            # Remove --single_stem to get both Vocals and Instrumental
            "--normalization", "0.9",
            "--mdxc_segment_size", "512",
            "--mdxc_overlap", "16",
            "--mdxc_batch_size", str(bs),
            "--use_soundfile",
        ]

    attempt_run_with_retry(
        build_roformer_cmd, GPU_BATCH_SIZE,
        description="Separating Stems"
    )

    # Find outputs
    candidates_vocals = list(separation_out_dir.glob(v_glob))
    candidates_back = list(separation_out_dir.glob(i_glob))

    if not candidates_vocals or not candidates_back:
        raise Exception("Separation failed to produce both stems.")

    return candidates_vocals[0], candidates_back[0]


def _enhance_vocals_step(vocals_wav, enhanced_vocals_dir, work_dir):
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

    cmd_enhance = [
        "resemble-enhance",
        str(enhance_input_dir),
        str(enhanced_vocals_dir),
        "--denoise_only",
        "--nfe", ENHANCE_NFE,
        "--solver", "rk4",
        "--tau", ENHANCE_TAU,
        "--device", CUDA_DEVICE
    ]

    run_command_with_progress(
        cmd_enhance, description="Enhancing Vocals"
    )

    try:
        shutil.rmtree(enhance_input_dir)
    except Exception:
        pass

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


def _denoise_music_step(music_wav, denoised_music_dir):
    """Step 4: Denoise Music (UVR-DeNoise-Lite)."""
    candidates_denoised = list(
        denoised_music_dir.glob("*.wav")
    )
    valid_denoised = [f for f in candidates_denoised if is_valid_audio(f)]

    if valid_denoised:
        log_msg("  [Step 3/5] Skipping Music Denoising (exists)")
        return valid_denoised[0]

    log_msg("  [Step 3/5] Denoising Music (UVR-DeNoise-Lite)...")

    def build_denoise_cmd(bs):
        return [
            "audio-separator", str(music_wav),
            "--model_filename", DENOISE_MODEL,
            "--output_dir", str(denoised_music_dir),
            "--output_format", "wav",
            "--single_stem", "No Noise",
            "--vr_batch_size", str(bs),
            "--vr_window_size", "320",
            "--use_soundfile",
        ]

    attempt_run_with_retry(
        build_denoise_cmd, GPU_BATCH_SIZE,
        description="Denoising Music (AI)"
    )

    candidates_denoised = list(
        denoised_music_dir.glob("*.wav")
    )

    clean_candidates = [
        f for f in candidates_denoised if "(No Noise)" in f.name
    ]

    if clean_candidates:
        result = clean_candidates[0]
    elif candidates_denoised:
        result = candidates_denoised[0]
    else:
        log_msg("    [Warning] UVR-DeNoise failed. Using raw music.",
                is_error=True)
        result = music_wav

    log_msg(f"    Selected Denoised Stem: {result.name}")
    return result


def _final_mix_step(
    video_path, enhanced_vocals_wav,
    denoised_music_wav, final_output_video
):
    """Step 5: Mix with FFmpeg."""
    log_msg("  [Step 5/5] Final Mix (32-bit Float)...")

    def build_mix_cmd(threads):
        return [
            FFMPEG_BIN, "-stats", "-hide_banner",
            "-threads", str(threads),
            "-i", str(video_path),          # 0: Video source
            "-i", str(enhanced_vocals_wav),
            "-i", str(denoised_music_wav),
            "-map", "0:v",
            "-filter_complex",
            f"[1:a]volume={VOCAL_MIX_VOL}[v];"
            f"[2:a]volume={MUSIC_MIX_VOL}[m];"
            "[v][m]amix=inputs=2:duration=longest:normalize=0[out]",
            "-map", "[out]",
            "-c:v", "copy",
            "-c:a", "pcm_f32le",
            "-shortest", "-y", str(final_output_video)
        ]

    # Calculate duration for progress bar
    duration = get_audio_duration_sec(enhanced_vocals_wav)

    attempt_cpu_run_with_retry(
        build_mix_cmd, CPU_THREADS,
        description="Final Mixing",
        total_duration=duration
    )


def _align_stems(original_wav, processed_wav, output_wav):
    """
    Step 4: Smart Audio Sync via Cross-Correlation.
    Calculates the lag between original and processed audio and shifts the processed
    audio to match the original's timing perfectly.
    """
    log_msg(f"  [Step 4/5] Sync: Aligning {processed_wav.name}...")

    try:
        # Load up to 60 seconds for correlation analysis (sufficient for offset detect)
        # We assume sample rates match (pipeline uses 44.1k everywhere)
        ref_audio, sr = sf.read(str(original_wav), frames=44100 * 60)
        proc_audio, _ = sf.read(str(processed_wav), frames=44100 * 60)

        # Convert stereo to mono for correlation
        if len(ref_audio.shape) > 1:
            ref_audio = np.mean(ref_audio, axis=1)
        if len(proc_audio.shape) > 1:
            proc_audio = np.mean(proc_audio, axis=1)

        # Pad shortest for calculation if needed
        n = len(ref_audio)
        m = len(proc_audio)
        if n == 0 or m == 0:
            log_msg("    [Warning] Audio empty, skipping sync.", is_error=True)
            # Ensure output is stereo even if skipping
            data, rate = sf.read(str(processed_wav), always_2d=True)
            if data.shape[1] == 1:
                data = np.tile(data, (1, 2))
            sf.write(str(output_wav), data, rate, subtype='FLOAT')
            return output_wav

        # Calculate Cross-Correlation using FFT (fastest)
        correlation = scipy.signal.correlate(ref_audio, proc_audio, mode='full', method='fft')
        lags = scipy.signal.correlation_lags(len(ref_audio), len(proc_audio), mode='full')
        lag = lags[np.argmax(correlation)]

        log_msg(f"    Detected Lag: {lag} samples ({lag / sr * 1000:.2f} ms)")

        if abs(lag) < 10:  # Negligible drift
            # Load full file to ensure stereo export
            full_proc_audio, sr = sf.read(str(processed_wav), always_2d=True)
            if full_proc_audio.shape[1] == 1:
                full_proc_audio = np.tile(full_proc_audio, (1, 2))  # Mono -> Stereo
            sf.write(str(output_wav), full_proc_audio, sr, subtype='FLOAT')
            return output_wav

        # Apply Shift to the FULL processed file
        full_proc_audio, sr = sf.read(str(processed_wav), always_2d=True)

        # Force Stereo (if mono)
        if full_proc_audio.shape[1] == 1:
            full_proc_audio = np.tile(full_proc_audio, (1, 2))

        # If Lag is Positive: processed is BEHIND original (needs to move earlier / cut start)
        # If Lag is Negative: processed is AHEAD of original (needs to move later / add silence)
        # Scipy definition: correlation[k] = sum_n ref[n] * proc[n+k]
        # Peak at +L means ref matches proc shifted by -L?
        # Actually simplest heuristic:
        # If peak is at lag > 0, it means Proc needs to be shifted 'right' (delayed) to match Ref?
        # Let's verify standard definition:
        # valid lag means ref[x] matches proc[x + lag].
        # So if lag is positive, proc is "earlier" in the array than matches in ref.
        # So we need to PAD proc with `lag` zeros to push it forward.

        if lag > 0:
            # Pad beginning with zeros
            pad = np.zeros((lag, full_proc_audio.shape[1]), dtype=full_proc_audio.dtype)
            shifted_audio = np.vstack((pad, full_proc_audio))
        else:
            # Cut beginning (lag is negative)
            cut = abs(lag)
            if cut >= len(full_proc_audio):
                shifted_audio = full_proc_audio  # Safety fallback
            else:
                shifted_audio = full_proc_audio[cut:]

        # Save aligned file
        sf.write(str(output_wav), shifted_audio, sr, subtype='FLOAT')
        return output_wav

    except Exception as e:
        log_msg(f"    [Warning] Sync failed ({e}). Using unaligned.", is_error=True)
        # Fallback with stereo enforcement
        try:
            data, rate = sf.read(str(processed_wav), always_2d=True)
            if data.shape[1] == 1:
                data = np.tile(data, (1, 2))
            sf.write(str(output_wav), data, rate, subtype='FLOAT')
        except Exception:  # pragma: no cover
            # Absolute worst case fallback
            shutil.copy(processed_wav, output_wav)

        return output_wav


def process_hybrid_audio(video_path, gpu_name, target_output_dir=OUTPUT_DIR):
    """Main entry point for processing high-fidelity restoration."""
    log_msg(f"\n[System] Processing Task: {video_path.name}")

    # Paths
    file_id = video_path.stem
    work_dir = TEMP_DIR / file_id
    work_dir.mkdir(parents=True, exist_ok=True)

    original_wav = work_dir / f"{video_path.stem}.wav"
    target_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_video = (
        target_output_dir / f"{video_path.stem}_Hybrid_Cleaned{video_path.suffix}"
    )

    if is_valid_audio(final_output_video):
        log_msg(f"  [System] Skipping: {video_path.name} (exists)")
        if work_dir.exists():  # pragma: no cover
            shutil.rmtree(work_dir, ignore_errors=True)
        return True

    try:
        # Define Paths
        separation_out_dir = work_dir / "separation"

        # 1. Extract High-Res Audio
        _extract_audio_step(video_path, original_wav)

        # 2. Split with BSRoformer (Single Pass)
        # Yields Vocals and Background (Instrumental)
        vocals_wav, background_wav = _separate_stems_step(
            original_wav, separation_out_dir
        )

        # Prepare Directories
        enhanced_vocals_dir = work_dir / "enhanced_vocals"
        denoised_music_dir = work_dir / "denoised_music"

        # 3 & 4. Enhance Vocals & Denoise Background
        # Can be run in parallel on ULTRA GPUs (using different models)
        if GPU_VRAM_GB >= 22:
            log_msg("  [System] ULTRA Profile: Parallel Enhancement/Denoising...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                f_enhance = executor.submit(
                    _enhance_vocals_step,
                    vocals_wav, enhanced_vocals_dir, work_dir
                )
                f_denoise = executor.submit(
                    _denoise_music_step,
                    background_wav, denoised_music_dir
                )

                enhanced_vocals_wav = f_enhance.result()
                denoised_music_wav = f_denoise.result()
        else:
            # Sequential Processing
            enhanced_vocals_wav = _enhance_vocals_step(
                vocals_wav, enhanced_vocals_dir, work_dir
            )
            denoised_music_wav = _denoise_music_step(
                background_wav, denoised_music_dir
            )

        # 5. Smart Audio Sync (Cross-Correlation)
        aligned_vocals = work_dir / f"aligned_{enhanced_vocals_wav.name}"
        aligned_music = work_dir / f"aligned_{denoised_music_wav.name}"

        _align_stems(original_wav, enhanced_vocals_wav, aligned_vocals)
        _align_stems(original_wav, denoised_music_wav, aligned_music)

        # 6. Final Mix
        _final_mix_step(
            video_path, aligned_vocals,
            aligned_music, final_output_video
        )

        shutil.rmtree(work_dir, ignore_errors=True)
        log_msg(f"  [System] Task Completed: {video_path.name}")
        return True

    except Exception as e:
        log_msg(f"  [Error] Processing failed: {e}", is_error=True)
        return False


def _show_banner(cpu_name, gpu_name):
    """Prints the application banner and hardware info."""
    print("=" * 60)
    print("   AI HYBRID VHS AUDIO RESTORER (OPTIMIZED) - v1.0.0")  # pragma: no cover
    print(f"   Running on: {platform.system()} {platform.release()}")  # pragma: no cover
    print("=" * 60 + "\n")  # pragma: no cover

    print("[HARDWARE DETECTED]")
    print(f"   CPU : {os.cpu_count()} Logical Cores ({cpu_name})")
    print(f"   GPU : {gpu_name} ({GPU_VRAM_GB:.2f} GB VRAM)\n")

    print(f"[AUTO-TUNED SETTINGS -> Profile: {PROFILE_NAME}]")
    print("   Audio Precision : 32-bit Float (WAV)")
    print("   AI Architecture : Hybrid (BS-Roformer + Lossless Background)")
    print(f"   Batch Size      : {GPU_BATCH_SIZE}")
    print(f"   Threads         : {CPU_THREADS}")
    print(f"   Mix Levels      : Vocals={VOCAL_MIX_VOL}, Music={MUSIC_MIX_VOL}")
    print(f"   Models          : {VOCALS_MODEL} / UVR-DeNoise")
    print(f"   Config Source   : {CONFIG_SOURCE}\n")


def _get_input_files():
    """Gathers input files from CLI args or interactive prompt."""
    files = []
    use_source_as_output = False

    if len(sys.argv) > 1:
        use_source_as_output = True
        print(f">> Arguments Detected: {len(sys.argv) - 1} items")
        for arg in sys.argv[1:]:
            path = Path(arg)
            if path.is_file() and path.suffix.lower() in EXTS:
                files.append(path)
            elif path.is_dir():
                print(f">> Scanning folder: {path.name}")
                files.extend([
                    f for f in path.iterdir()
                    if f.is_file() and f.suffix.lower() in EXTS
                ])
    else:
        try:
            # Matches user report: "Please Drag & Drop a video file here and press Enter"
            print(">> Please Drag & Drop a video file here and press Enter:")
            user_input = input(">>Path: ").strip()

            # Clean input handles:
            # 1. PowerShell Drag & Drop (& 'path')
            # 2. Terminal file:// prefix
            # 3. Surrounding quotes

            if user_input.startswith("&"):
                user_input = user_input[1:].strip()

            if user_input.lower().startswith("file://"):
                user_input = user_input[7:].strip()

            if user_input.startswith('"') and user_input.endswith('"'):
                user_input = user_input[1:-1]
            elif user_input.startswith("'") and user_input.endswith("'"):
                user_input = user_input[1:-1]

            if not user_input:
                print(">> Interactive Mode: Drag & Drop files or press "
                      "Enter to scan 'input' folder.")
                print(">> Scanning 'input' folder...")
                INPUT_DIR.mkdir(exist_ok=True)
                files = [
                    f for f in INPUT_DIR.iterdir()
                    if f.suffix.lower() in EXTS
                ]
            else:
                path = Path(user_input)
                if not path.exists():
                    print(f">> [Error] File not found: {path}")
                    return [], False

                if path.is_file():
                    if path.suffix.lower() in EXTS:
                        print(f">> Selected File: {path.name}")
                        files.append(path)
                        use_source_as_output = True
                    else:
                        print(f">> [Error] Unsupported extension: {path.suffix}")
                        print(f">> Supported: {EXTS}")
                elif path.is_dir():
                    print(f">> Scanning folder: {path.name}")
                    files = [
                        f for f in path.iterdir()
                        if f.is_file() and f.suffix.lower() in EXTS
                    ]
                    use_source_as_output = True
                else:
                    print(f">> [Error] Path is not a file or directory: {path}")
        except (EOFError, KeyboardInterrupt):
            pass

    return files, use_source_as_output


def main():
    import time
    print("\n[AI ENGINE INITIALIZATION]")

    draw_progress_bar(10, "Initializing Core Systems...")
    time.sleep(0.3)
    draw_progress_bar(30, "Scanning Hardware...")
    time.sleep(0.2)

    cpu_name = get_cpu_name()
    draw_progress_bar(45, f"CPU Detected: {os.cpu_count()} Cores")

    gpu_name = get_gpu_name()
    draw_progress_bar(60, f"GPU Detected: {gpu_name}")

    p_name = PROFILE_NAME.split('(')[0].strip()
    draw_progress_bar(75, f"Applying Optimization Profile: {p_name}")
    time.sleep(0.2)

    if not check_dependencies():
        print("\n[Init] Critical Error: Dependencies Missing.")
        return

    draw_progress_bar(90, "Verifying Libraries...")
    time.sleep(0.2)
    draw_progress_bar(100, "Initialization Complete.")
    time.sleep(0.4)

    _show_banner(cpu_name, gpu_name)

    print("-" * 60)
    print(" [HOW TO USE]")
    print(" 1. Drag and Drop a video file (or folder) here.")
    print(" 2. Or paste the file path below.")
    print("-" * 60 + "\n")

    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    files, use_source_as_output = _get_input_files()

    if not files:
        print(">> No valid video files found.")
        return

    print(f"\n[System] Found {len(files)} files in queue.")
    print("[System] Starting Batch Processing...")

    for video_path in files:
        target_dir = video_path.parent if use_source_as_output else OUTPUT_DIR
        process_hybrid_audio(
            video_path, gpu_name, target_output_dir=target_dir
        )

    print("\n" + "=" * 60)
    print("   BATCH PROCESSING COMPLETE")
    print("=" * 60)
    try:
        input("Press Enter to exit...")
    except (EOFError, KeyboardInterrupt):  # pragma: no cover
        pass


if __name__ == "__main__":
    main()
