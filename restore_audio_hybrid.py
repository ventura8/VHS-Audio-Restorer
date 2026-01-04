import os
import subprocess
import shutil
import sys
import platform
import datetime
from pathlib import Path
import re
import threading

from typing import Set, Optional, Any
import signal
import atexit
import time
import concurrent.futures
import collections

import soundfile as sf  # type: ignore
import torch
import yaml
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

# Modern terminal support
try:
    from colorama import init
    init()
except ImportError:
    pass

# Optional imports for DTW
DTW_IMPORT_ERROR: Optional[str] = None
try:
    import librosa
    import fastdtw
except ImportError as e:
    # We will log this later in check_dependencies or when used,
    # but storing the error helps debugging.
    DTW_IMPORT_ERROR = str(e)
    librosa = None  # type: ignore
    fastdtw = None  # type: ignore
else:
    # DTW_IMPORT_ERROR is already None, but explicit assignment is fine or we can pass
    pass

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
        "background_mix_volume": 1.0,
        "extensions": ['.mp4', '.mkv', '.avi', '.mov'],
        "vocals_model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "background_model": "UVR-MDX-NET-Inst_HQ_3.onnx",
        "denoise_model": "UVR-DeNoise-Lite.pth",
        "enhance_nfe": 128,
        "enhance_tau": 0.5,
        "sync_method": "dtw",    # 'check' or 'dtw'
        "dtw_resolution": 40,    # Analysis resolution in Hz (40Hz = 25ms, Sufficient for Lipsync)
        "process_mode": "hybrid",  # 'hybrid' (default) or 'denoise_only'
        "debug_logging": False
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
LOG_FILE = Path("session_log.txt")

EXTS = set(CONFIG["extensions"])
KEEP_INPUT_FILES = os.environ.get("AI_RESTORE_TEST_MODE") == "1"

# Audio mix levels
VOCAL_MIX_VOL = CONFIG["vocal_mix_volume"]
BACKGROUND_MIX_VOL = CONFIG["background_mix_volume"]

# AI Configs
VOCALS_MODEL = CONFIG["vocals_model"]
BACKGROUND_MODEL = CONFIG["background_model"]
DENOISE_MODEL = CONFIG["denoise_model"]
ENHANCE_NFE = str(CONFIG["enhance_nfe"])
ENHANCE_NFE = str(CONFIG["enhance_nfe"])
ENHANCE_TAU = str(CONFIG["enhance_tau"])
SYNC_METHOD = CONFIG["sync_method"]
DTW_RESOLUTION = int(CONFIG["dtw_resolution"])
PROCESS_MODE = CONFIG["process_mode"]
DEBUG_LOGGING = CONFIG.get("debug_logging", False)


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
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


# Global Thread Locks and State
_print_lock = threading.Lock()
_last_bar_time = 0
_last_bar_pc = -1.0


def _build_progress_info(percent, elapsed_sec, media_sec, total_duration=None):
    """
    Builds the info string: 74.3% | 00:20:43,400 / 00:27:52,777 | ETA 00:00:39 | 10.76x
    """
    # 1. Percent
    parts = [f"{percent:5.1f}%"]

    # 2. Time / Total
    if media_sec is not None:
        current_str = format_time(media_sec)
        if total_duration:
            total_str = format_time(total_duration)
            parts.append(f"{current_str} / {total_str}")
        else:
            parts.append(f"{current_str}")
    elif elapsed_sec is not None:
        # Fallback to elapsed if no media time (e.g. non-ffmpeg steps)
        parts.append(format_time(elapsed_sec))

    # 3. ETA
    if percent > 0 and percent < 100 and elapsed_sec is not None:
        total_est = (elapsed_sec / percent) * 100
        eta_sec = total_est - elapsed_sec
        parts.append(f"ETA {format_time(eta_sec).split(',')[0]}")  # Keep ETA simple (seconds)

    # 4. Speed
    if media_sec is not None and elapsed_sec is not None and elapsed_sec > 0.1:
        speed = media_sec / elapsed_sec
        parts.append(f"{speed:5.2f}x")

    return " | ".join(parts)


def _adjust_bar_layout(width, info_str, label, columns):
    """Adjusts bar width and truncates label to fit terminal."""
    overhead = 8  # Indent(4) + [] + Space + Safety(1)
    clean_label = re.sub(r'[\r\n]', '', label).strip()

    label_len = len(clean_label) if clean_label else 0
    total_len = overhead + width + len(info_str) + label_len + 3  # +3 for '   ' padding

    if total_len > columns:
        # Strategy 1: Truncate Label first (prefer keeping the bar)
        excess = total_len - columns
        if label_len > 20:
            # How much can we take from label?
            # We want to keep at least 15 chars of label
            label_can_give = max(0, label_len - 15)
            shrink_label = min(excess, label_can_give)
            label_len -= shrink_label
            clean_label = clean_label[:label_len] + "..."
            total_len -= (shrink_label - 3)  # account for ellipsis

        if total_len > columns:
            # Strategy 2: Shrink Bar
            excess = total_len - columns
            can_shrink = max(0, width - 5)  # Keep bar at least 5 wide
            shrink_amt = min(excess, can_shrink)
            width -= shrink_amt
            total_len -= shrink_amt

        if total_len > columns:
            # Strategy 3: Hard Truncate Label
            excess = total_len - columns
            label_available = max(0, len(clean_label) - excess - 3)
            if label_available < 5:
                clean_label = ""
            else:
                clean_label = clean_label[:label_available] + "..."

    return width, info_str, clean_label


def _get_terminal_columns(default=79):
    """Returns terminal columns with safety margin."""
    try:
        # Remove 80-column cap for modern terminals
        return shutil.get_terminal_size((80, 20)).columns - 1
    except Exception:
        return default


def _draw_bar_line(width, filled_length, info_str, label=""):
    """Draws the final bar line with explicit clearing and no-wrap safety."""
    bar = "█" * filled_length + "░" * (width - filled_length)

    # Standardized 3-space padding
    if label:
        line_content = f"   {label}[{bar}] {info_str}"
    else:
        line_content = f"   [{bar}] {info_str}"

    # Terminal width safety
    cols = _get_terminal_columns()
    if len(line_content) >= cols:
        line_content = line_content[:cols - 1]

    with _print_lock:
        # \r to start, \033[K to clear, then content. NO trailing newline.
        sys.stdout.write(f"\r\033[K{line_content}")
        sys.stdout.flush()


def draw_progress_bar(
    percent, label="", width=20, elapsed_sec=None, media_sec=None, total_duration=None
):
    """
    Draws a modern visual progress bar with rate-limiting.
    """
    global _last_bar_time, _last_bar_pc

    percent = max(0.0, min(100.0, float(percent)))
    now = time.time()

    # Rate limit: Max 20 FPS, but always allow 0%, 100%, or major jumps
    if now - _last_bar_time < 0.05 and abs(percent - _last_bar_pc) < 1.0 and 0 < percent < 100:
        return

    _last_bar_time = now
    _last_bar_pc = percent

    columns = _get_terminal_columns()

    # Build Info String
    info_str = _build_progress_info(percent, elapsed_sec, media_sec, total_duration)

    # Clean label and ensure it has spacing if present
    clean_label = re.sub(r'[\r\n]', '', label).strip()

    # Layout Adjustment
    width, info_str, final_label = _adjust_bar_layout(width, info_str, clean_label, columns)

    filled_length = int(width * percent // 100)
    if width < 2:
        width = 2

    _draw_bar_line(width, filled_length, info_str, final_label)


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

    # DEBUG logs never print to console unless debug_logging is enabled
    should_print = console
    if effective_level == "DEBUG" and not DEBUG_LOGGING:
        should_print = False

    if should_print:
        # Clear any active progress bar line before printing log
        # Use 3 spaces for all log messages too
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        print(f"   {message}" if not message.startswith("   ") else message)

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
    Returns False if corrupted, empty, or extremely short (<0.1s).
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
            if f.frames > 0 and f.samplerate > 0:
                duration = f.frames / f.samplerate
                if duration > 0.1:  # Must be at least 100ms
                    return True
    except Exception:
        pass

    return False


def _save_audio_atomic(file_path, data, sample_rate, subtype='FLOAT'):
    """
    Saves audio to a temporary file, then renames it to the final path.
    This prevents corrupted partial files if the process is interrupted.
    """
    path = Path(file_path)
    # Use .tmp.wav to ensure soundfile and is_valid_audio recognize the format
    temp_path = path.with_suffix(f".tmp{path.suffix}")

    try:
        sf.write(str(temp_path), data, sample_rate, subtype=subtype)

        # Verify written file is valid before renaming
        if is_valid_audio(temp_path):
            if path.exists():
                path.unlink()
            temp_path.rename(path)
            return True
        else:
            log_msg(f"[Error] Atomic Save Failed: {temp_path} is invalid.", is_error=True)
            if temp_path.exists():
                temp_path.unlink()
            return False

    except Exception as e:
        log_msg(f"[Error] Failed to save audio {path}: {e}", is_error=True)
        if temp_path.exists():
            temp_path.unlink()
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


def get_video_duration_sec(file_path):
    """Returns duration of video file in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
        ]
        output = subprocess.check_output(cmd, universal_newlines=True).strip()
        return float(output)
    except Exception:  # pragma: no cover
        # Fallback to audio duration if ffprobe fails
        return get_audio_duration_sec(file_path)


def parse_ffmpeg_time(line):
    """Extracts time=HH:MM:SS.mm from FFmpeg output."""
    match = re.search(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})", line)
    if match:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 100.0
    return None


def _parse_tqdm_progress(line, tqdm_re):
    """Parses TQDM progress lines, returning only the percentage for our own bar."""
    match = tqdm_re.search(line)
    if not match:
        return None, None

    return float(match.group(1)), ""


def _monitor_process_output(process, start_time, duration, description, tqdm_re):
    """Monitors process stdout for progress updates."""
    output_buffer = collections.deque(maxlen=20)

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break

        if line:
            output_buffer.append(line)
            # log_msg(f"READ LINE: {line.strip()}", level="DEBUG") # For extreme debugging

            # 1. Check for FFmpeg 'time='
            current_time = parse_ffmpeg_time(line)
            if current_time is not None and duration:
                percent = (current_time / duration) * 100
                elapsed = time.time() - start_time
                draw_progress_bar(
                    percent, description,
                    elapsed_sec=elapsed, media_sec=current_time,
                    total_duration=duration
                )
                continue

            # 2. Check for TQDM '%'
            if "muxing overhead" in line.lower():
                continue

            percent, tqdm_info = _parse_tqdm_progress(line, tqdm_re)
            if percent is not None:
                elapsed = time.time() - start_time
                media_sec = (percent / 100) * duration if duration else None

                # Filter updates - only print if percent changed by at least 0.1%
                # or if it's the 100% or 0% mark
                if (not hasattr(process, '_last_pc') or
                    abs(percent - process._last_pc) >= 0.1 or
                        percent in [0, 100]):

                    process._last_pc = percent
                    label = f"{description} {tqdm_info}" if tqdm_info else description
                    draw_progress_bar(
                        percent, label,
                        elapsed_sec=elapsed, media_sec=media_sec
                    )
    return output_buffer


def run_command_with_progress(
    cmd, env=None, description="Running...", total_duration=None
):
    """
    Runs a subprocess and parses progress from its output.
    Supports FFmpeg and TQDM-style output.
    """
    start_time = time.time()
    sys.stdout.write("\n")  # Ensure we start on a new line
    sys.stdout.flush()
    if env is None:
        env = os.environ.copy()
    else:
        env = env.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace"
    )
    _active_processes.add(process)

    duration = max(0.1, total_duration) if total_duration else None
    # Stricter TQDM regex to avoid false positives and handle both | and : styles
    tqdm_re = re.compile(r'(\d+)%\s*[|:]')

    try:
        output_buffer = _monitor_process_output(
            process, start_time, duration, description, tqdm_re
        )
    finally:
        _active_processes.discard(process)

    process.wait()

    if process.returncode == 0:
        elapsed = time.time() - start_time
        draw_progress_bar(
            100.0, description,
            elapsed_sec=elapsed, media_sec=duration
        )
        sys.stdout.write("\n")
    else:
        sys.stdout.write("\n")
        log_msg(f"\n[Error] Command {cmd[0]} failed. Last output:", is_error=True)
        for err_line in output_buffer:
            log_msg(f"  > {err_line.strip()}", is_error=True)

        raise subprocess.CalledProcessError(process.returncode, cmd)


def attempt_run_with_retry(
    command_builder_func, initial_batch_size, description="Running...",
    total_duration=None
):
    """
    Retries a command with reduced GPU batch sizes on OOM.
    command_builder_func: Accepts 'batch_size' (int), returns [cmd, args...].
    """
    current_batch_size = initial_batch_size

    while True:
        try:
            cmd = command_builder_func(current_batch_size)
            # Use run_command_with_progress to get %, ETA, and Speed
            run_command_with_progress(
                cmd,
                description=f"{description} (BS:{current_batch_size})",
                total_duration=total_duration
            )
            return True  # Success

        except subprocess.CalledProcessError as e:
            if current_batch_size > 1:
                log_msg(
                    "    [Warning] GPU failed. Retrying with reduced batch...",
                    is_error=True
                )
                if torch.cuda.is_available():  # pragma: no cover
                    torch.cuda.empty_cache()
                import gc  # pragma: no cover
                gc.collect()
                current_batch_size = max(1, current_batch_size // 2)
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


def _extract_audio_step(video_path, original_wav, total_duration=None):
    """Step 1: Extract High-Res Audio."""
    if is_valid_audio(original_wav):
        log_msg("  [System] Skipping Extraction (valid exists)")
    else:
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
        "--nfe", ENHANCE_NFE,
        "--solver", "rk4",
        "--tau", ENHANCE_TAU,
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


def is_valid_video(file_path):
    """
    Checks if a video file exists and has reasonable size.
    Does not do full header check to avoid overhead, but filters empty files.
    """
    path = Path(file_path)
    if not path.exists():
        return False
    # 1MB minimum to be considered a successful video write
    if path.stat().st_size < 1024 * 1024:
        return False
    return True


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
    tmp_output = final_output_video.with_suffix(".tmp.mp4")

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


def _apply_warp_gpu(audio_np, source_indices_np):
    """
    Applies warping using GPU acceleration (torch.nn.functional.grid_sample).
    Audio: (samples, channels) numpy
    Indices: (output_samples,) numpy - The source index for each output sample.
    """
    try:
        if not torch.cuda.is_available():
            return None

        device = torch.device("cuda")

        # Prepare Audio: (N, C, H, W) -> (1, Channels, 1, Samples)
        # Audio is currently (Samples, Channels) from soundfile
        # Transpose to (Channels, Samples)
        audio_t = torch.from_numpy(audio_np.T).float().unsqueeze(0).unsqueeze(2).to(device)

        # Normalize Indices to [-1, 1] for grid_sample
        # -1 = 0, +1 = MaxSample
        W_in = audio_np.shape[0]
        # Avoid division by zero
        if W_in <= 1:
            return None

        indices_t = torch.from_numpy(source_indices_np).float().to(device)
        grid_x = 2.0 * indices_t / (W_in - 1) - 1.0

        # Construct Grid: (N, H_out, W_out, 2)
        # Here H_out=1, W_out=LenIndices
        W_out = len(source_indices_np)

        # Stack (x, y) coordinates
        # y is always 0 (center of 1-pixel height)
        # Shape: (1, 1, W_out, 2)
        grid = torch.zeros(1, 1, W_out, 2, device=device)
        grid[0, 0, :, 0] = grid_x
        grid[0, 0, :, 1] = 0

        # Resample
        # align_corners=True ensures -1 maps to index 0 and 1 maps to index W-1
        warped = torch.nn.functional.grid_sample(
            audio_t, grid,
            mode='bicubic',
            padding_mode='zeros',
            align_corners=True
        )

        # Output: (1, C, 1, W_out)
        # Back to (W_out, C)
        warped_np = warped.squeeze(2).squeeze(0).permute(1, 0).cpu().numpy()
        return warped_np

    except Exception as e:
        log_msg(f"    [Warning] GPU Warp failed ({e}). Fallback to CPU.")
        return None


# ---------------------------------------------------------
# Parallel DTW Worker & Helpers
# ---------------------------------------------------------
def _run_fastdtw_chunk(args):
    """
    Worker function for Parallel DTW.
    Args: (ref_segment, proc_segment, radius)
    Returns: path (list of [ref_idx, proc_idx])
    NOTE: 'dist' function (euclidean) must be imported in worker scope or passed.
    SciPy euclidean is picklable.
    """
    ref_seg, proc_seg, radius = args
    # Ensure dependencies in worker process
    import fastdtw
    from scipy.spatial.distance import euclidean

    _, path = fastdtw.fastdtw(ref_seg, proc_seg, radius=radius, dist=euclidean)
    return path


def _run_gpu_dtw_chunk(args):
    """
    Worker function for GPU-accelerated DTW (Hybrid).
    1. Compute Euclidean Distance Matrix on GPU (torch.cdist).
    2. Compute DTW Path on CPU (librosa.sequence.dtw).

    Args: (ref_seg, proc_seg, radius)
    Returns: path (list of [ref_idx, proc_idx])
    """
    ref_seg, proc_seg, radius = args

    # 1. GPU Distance Calculation
    import torch
    import librosa

    # Ensure tensors are on GPU
    # Note: ref_seg/proc_seg are numpy arrays (Features x Time) -> Need Transpose?
    # ref_seg passed in is (Time, Features) from `chunks.append`.
    # torch.cdist expects (B, P, M) or (P, M).

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_t = torch.tensor(ref_seg, dtype=torch.float32, device=device)
    proc_t = torch.tensor(proc_seg, dtype=torch.float32, device=device)

    # Compute Cost Matrix (N x M)
    # torch.cdist computes Euclidean distance
    cost_matrix = torch.cdist(ref_t, proc_t).cpu().numpy()

    # 2. CPU Path Finding (Librosa/Numba)
    # librosa.sequence.dtw expects 'C' as pre-computed cost matrix.
    # It returns D (cumulative cost) and wp (path).
    # wp is (N_steps, 2) in (row, col) -> (ref_idx, proc_idx)
    # Librosa returns path from end-to-start ([N-1, M-1] ... [0, 0]).

    _, wp = librosa.sequence.dtw(C=cost_matrix, global_constraints=False)

    # Reverse to get Start -> End ([0, 0] ... [N-1, M-1])
    wp = wp[::-1]

    # Convert to list of lists for compatibility
    return wp.tolist()


def _load_dtw_features(original_wav, processed_wav):
    """Loads audio and computes chroma features for DTW."""
    ANALYSIS_SR = 8192
    hop_length = int(ANALYSIS_SR / DTW_RESOLUTION)

    t0 = time.time()
    ref_y, _ = librosa.load(str(original_wav), sr=ANALYSIS_SR, mono=True)
    proc_y, _ = librosa.load(str(processed_wav), sr=ANALYSIS_SR, mono=True)
    log_msg(f"    Loaded for analysis in {time.time() - t0:.2f}s")

    audio_dur = len(ref_y) / ANALYSIS_SR

    ref_chroma = librosa.feature.chroma_stft(
        y=ref_y, sr=ANALYSIS_SR, hop_length=hop_length, n_fft=2048
    )
    proc_chroma = librosa.feature.chroma_stft(
        y=proc_y, sr=ANALYSIS_SR, hop_length=hop_length, n_fft=2048
    )

    return ref_chroma.T, proc_chroma.T, audio_dur, hop_length, ANALYSIS_SR


def _prepare_dtw_chunks(ref_features, proc_features):
    """Splits features into overlapping chunks."""
    radius = max(5, int(DTW_RESOLUTION * 0.3))

    # Chunk size: 3000 frames @ 40Hz = 75 seconds.
    CHUNK_SIZE = 3000
    OVERLAP = 100

    len_ref = len(ref_features)
    chunks = []
    chunk_starts = []

    for i in range(0, len_ref, CHUNK_SIZE - OVERLAP):
        start = i
        end = min(len_ref, i + CHUNK_SIZE)
        ref_seg = ref_features[start:end]
        proc_seg = proc_features[start:end]
        chunks.append((ref_seg, proc_seg, radius))
        chunk_starts.append(start)
        if end == len_ref:
            break

    return chunks, chunk_starts


def _execute_parallel_dtw(chunks):
    """Executes DTW chunks in parallel on CPU or GPU."""
    use_gpu_dtw = False
    try:
        if torch.cuda.is_available() and GPU_VRAM_GB >= 4:
            use_gpu_dtw = True
    except Exception:
        pass

    if use_gpu_dtw:
        max_workers = min(len(chunks), 4)
        Worker = _run_gpu_dtw_chunk
        Executor = concurrent.futures.ThreadPoolExecutor
        log_msg(f"    GPU Optimization ENABLED (VRAM: {GPU_VRAM_GB}GB). Using Torch+Librosa.")
    else:
        max_workers = min(len(chunks), CPU_THREADS)
        Worker = _run_fastdtw_chunk
        Executor = concurrent.futures.ProcessPoolExecutor

    log_msg(f"    Spawning {max_workers} workers for {len(chunks)} chunks...")

    results_map = {}
    chunks_done = 0
    num_chunks = len(chunks)
    t0 = time.time()

    draw_progress_bar(0, "DTW Sync: Starting...")

    with Executor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(Worker, chunk): i
            for i, chunk in enumerate(chunks)
        }

        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                res = future.result()
                results_map[chunk_idx] = res
            except Exception as e:
                log_msg(f"    Chunk {chunk_idx} failed: {e}", is_error=True)
                raise e

            chunks_done += 1
            percent = (chunks_done / num_chunks) * 100
            elapsed = time.time() - t0
            draw_progress_bar(
                percent, f"DTW Sync: Chunk {chunks_done}/{num_chunks}",
                elapsed_sec=elapsed
            )

    path_segments = [results_map[i] for i in range(num_chunks)]
    sys.stdout.write("\n")
    return path_segments


def _stitch_dtw_path(path_segments, chunk_starts):
    """Stitches chunked paths into a single monotonic path."""
    full_path = []

    for i, segment_path in enumerate(path_segments):
        base_idx = chunk_starts[i]
        seg_arr = np.array(segment_path)
        seg_arr += base_idx
        full_path.append(seg_arr)

    path = np.vstack(full_path)

    # Sort by Ref Index (Time)
    sort_order = np.argsort(path[:, 0])
    path = path[sort_order]

    # Unique Ref indices
    _, unique_mask = np.unique(path[:, 0], return_index=True)
    path = path[unique_mask]

    # Enforce Monotonicity on Proc Indices
    path[:, 1] = np.maximum.accumulate(path[:, 1])

    log_msg(f"    Stitched segments into {len(path)} points.")
    return path


def _warp_aligned_audio_cpu(full_proc_audio, source_indices, num_channels, output_wav, full_sr):
    """Fallback CPU warping."""
    log_msg("    GPU Warp unavailable/failed. Using CPU (scipy)...")
    out_audio = np.zeros_like(full_proc_audio)
    for ch in range(num_channels):
        out_audio[:, ch] = map_coordinates(
            full_proc_audio[:, ch],
            [source_indices],
            order=3
        )
    _save_audio_atomic(output_wav, out_audio, full_sr, subtype='FLOAT')


def _warp_aligned_audio(processed_wav, output_wav, path, ref_features_len, proc_features_len):
    """Warps the processed audio to align with reference."""

    # Coordinate Mapping Function
    ref_indices = path[:, 0]
    proc_indices = path[:, 1]
    _, unique_idxs = np.unique(ref_indices, return_index=True)
    unique_idxs.sort()
    target_t = ref_indices[unique_idxs]
    source_t = proc_indices[unique_idxs]

    warp_func = interp1d(target_t, source_t, kind='linear', fill_value="extrapolate")

    full_proc_audio, full_sr = sf.read(str(processed_wav), always_2d=True)
    num_frames = len(full_proc_audio)
    num_channels = full_proc_audio.shape[1]

    max_ref_idx = ref_features_len - 1
    max_proc_idx = proc_features_len - 1

    # Map: Full Sample Index -> Feature Index
    grid_in_feature_domain = np.linspace(0, max_ref_idx, num_frames)
    warped_feature_indices = warp_func(grid_in_feature_domain)

    # Smoothing
    # feature_rate = full_sr / (full_sr / DTW_RESOLUTION) # approx
    # Actually just use hardcoded reasonable window if hop unknown or pass it
    # We passed features lengths so we can estimate if we want but
    # lets assume ~86Hz default.
    window_length = 51  # minimal default
    if len(warped_feature_indices) > window_length:
        try:
            from scipy.signal import savgol_filter
            warped_feature_indices = savgol_filter(warped_feature_indices, window_length, 3)
        except Exception:
            pass

    # Feature Index -> Full Sample Index
    source_indices = warped_feature_indices * (num_frames / max_proc_idx)

    t0 = time.time()
    warped_gpu = _apply_warp_gpu(full_proc_audio, source_indices)

    if warped_gpu is not None:
        log_msg(f"    GPU Warping complete in {time.time() - t0:.2f}s")
        _save_audio_atomic(output_wav, warped_gpu, full_sr, subtype='FLOAT')
    else:
        _warp_aligned_audio_cpu(full_proc_audio, source_indices, num_channels, output_wav, full_sr)

    draw_progress_bar(100, "Sync: Complete")
    sys.stdout.write("\n")


def _align_stems_dtw(original_wav, processed_wav, output_wav):
    """
    Advanced Sync: Dynamic Time Warping (DTW).
    Corrects variable drift (wow/flutter) by warping the processed audio
    to match the original timing perfectly.
    """
    if librosa is None or fastdtw is None:
        msg = "[Warning] DTW dependencies missing. Falling back to Shift."
        if DTW_IMPORT_ERROR:
            msg += f" (Error: {DTW_IMPORT_ERROR})"
        log_msg(msg, is_error=True)
        return _align_stems_shift(original_wav, processed_wav, output_wav)

    log_msg(f"    DTW Sync: Warping {processed_wav.name}...")

    try:
        ref_features, proc_features, audio_dur, _, _ = _load_dtw_features(original_wav, processed_wav)

        res_factor = DTW_RESOLUTION / 100.0
        est_dtw_time = audio_dur * 0.06 * res_factor
        log_msg(f"    Computing DTW Path (Est. Time: {format_time(est_dtw_time)})...")

        chunks, chunk_starts = _prepare_dtw_chunks(ref_features, proc_features)

        path_segments = _execute_parallel_dtw(chunks)
        log_msg("    DTW Path computed, stitching...")

        path = _stitch_dtw_path(path_segments, chunk_starts)

        _warp_aligned_audio(processed_wav, output_wav, path, len(ref_features), len(proc_features))

        return output_wav

    except Exception as e:
        log_msg(f"    [Warning] DTW Sync failed ({e}). Fallalng back to Shift.", is_error=True)
        return _align_stems_shift(original_wav, processed_wav, output_wav)


def _align_stems(original_wav, processed_wav, output_wav):
    if SYNC_METHOD == "dtw":
        return _align_stems_dtw(original_wav, processed_wav, output_wav)
    else:
        return _align_stems_shift(original_wav, processed_wav, output_wav)


def _calculate_cross_correlation_lag(ref_audio, proc_audio, sr):
    """Calculates lag using cross correlation."""
    # Convert stereo to mono for correlation
    if len(ref_audio.shape) > 1:
        ref_audio = np.mean(ref_audio, axis=1)
    if len(proc_audio.shape) > 1:
        proc_audio = np.mean(proc_audio, axis=1)

    draw_progress_bar(50, "Sync: Calculating Correlation...")
    correlation = scipy.signal.correlate(ref_audio, proc_audio, mode='full', method='fft')
    lags = scipy.signal.correlation_lags(len(ref_audio), len(proc_audio), mode='full')
    lag = lags[np.argmax(correlation)]

    log_msg(f"    Detected Lag: {lag} samples ({lag / sr * 1000:.2f} ms)")
    return lag


def _apply_shift_to_audio(processed_wav, output_wav, lag):
    """Applies shift to audio and saves it."""
    draw_progress_bar(80, "Sync: Applying Shift...")

    proc_audio, proc_sr = sf.read(str(processed_wav), always_2d=True)
    if proc_audio.shape[1] == 1:
        proc_audio = np.tile(proc_audio, (1, 2))

    shift_samples = -lag

    if shift_samples != 0:
        log_msg(f"    Applying shift: {shift_samples} samples")
        shifted_audio = np.roll(proc_audio, shift_samples, axis=0)

        # Zero out wrapped around part to avoid artifacts
        if shift_samples > 0:
            shifted_audio[:shift_samples] = 0
        else:
            shifted_audio[shift_samples:] = 0
    else:
        log_msg("    No shift needed.")
        shifted_audio = proc_audio

    # Atomic Write
    if not _save_audio_atomic(output_wav, shifted_audio, proc_sr, subtype='FLOAT'):
        raise Exception(f"Alignment Failed: Could not save shifted audio to {output_wav}")


def _align_stems_shift(original_wav, processed_wav, output_wav):
    """
    Step 4: Smart Audio Sync via Cross-Correlation.
    Calculates the lag between original and processed audio and shifts the processed
    audio to match the original timing perfectly.
    """
    log_msg(f"    Shift Sync: Aligning {processed_wav.name}...")

    try:
        draw_progress_bar(10, "Sync: Loading Audio...")
        ref_audio, sr = sf.read(str(original_wav), frames=44100 * 60)
        proc_audio, _ = sf.read(str(processed_wav), frames=44100 * 60)

        draw_progress_bar(30, "Sync: Preparing Stems...")

        n = len(ref_audio)
        m = len(proc_audio)
        if n == 0 or m == 0:
            log_msg("    [Warning] Audio empty, skipping sync.", is_error=True)
            data, rate = sf.read(str(processed_wav), always_2d=True)
            if data.shape[1] == 1:
                data = np.tile(data, (1, 2))
            sf.write(str(output_wav), data, rate, subtype='FLOAT')
            draw_progress_bar(100, "Sync: Skipped (Empty)")
            sys.stdout.write("\n")
            return output_wav

        lag = _calculate_cross_correlation_lag(ref_audio, proc_audio, sr)
        _apply_shift_to_audio(processed_wav, output_wav, lag)

        draw_progress_bar(100, "Sync: Complete")
        sys.stdout.write("\n")
        return output_wav

    except Exception as e:
        log_msg(f"    [Warning] Sync failed ({e}). Using unaligned.", is_error=True)
        try:
            data, rate = sf.read(str(processed_wav), always_2d=True)
            if data.shape[1] == 1:
                data = np.tile(data, (1, 2))
            sf.write(str(output_wav), data, rate, subtype='FLOAT')
        except Exception:
            shutil.copy(processed_wav, output_wav)

        return output_wav


def process_hybrid_audio(video_path, gpu_name, target_output_dir=OUTPUT_DIR):
    """Main entry point for processing high-fidelity restoration."""
    log_msg(f"\n[System] Processing Task: {video_path.name}")

    # Paths
    file_id = video_path.stem
    # Use hidden temp folder in the input directory
    work_dir = video_path.parent / f".temp_work_{file_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    original_wav = work_dir / f"{video_path.stem}.wav"
    target_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_video = (
        target_output_dir / f"{video_path.stem}_Hybrid_Cleaned{video_path.suffix}"
    )

    if is_valid_video(final_output_video):
        log_msg(f"  [System] Skipping: {video_path.name} (exists)")
        if work_dir.exists():
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
        return True

    try:
        video_dur = get_video_duration_sec(video_path)
        log_msg(f"  [System] Video Duration: {format_time(video_dur)}")

        if PROCESS_MODE == "denoise_only":
            log_msg("  [System] Mode: Denoise Only (Skipping Separation)")

            # 1. Extract
            _extract_audio_step(video_path, original_wav, total_duration=video_dur)

            denoised_dir = work_dir / "denoised_full"
            denoised_dir.mkdir(parents=True, exist_ok=True)

            # 2. Denoise the FULL file using the Background Denoising Model (UVR-DeNoise)
            # We reuse _denoise_background_step as logic is identical
            denoised_wav = _denoise_background_step(
                original_wav, denoised_dir, total_duration=video_dur
            )

            # 3. Sync (Denoising can introduce slight latency, though usually negligible)
            # We still run sync to catch any processing drift if enabled
            aligned_wav = work_dir / f"aligned_{denoised_wav.name}"
            _align_stems(original_wav, denoised_wav, aligned_wav)

            # 4. Remux (Simple Mix)
            # Use final mix but just map the single audio track
            log_msg("  [System] Remuxing Denoised Audio...")

            # Simplified FFmpeg command for single track remux
            cmd_remux = [
                FFMPEG_BIN, "-stats", "-hide_banner",
                "-threads", str(CPU_THREADS),
                "-i", str(video_path),
                "-i", str(aligned_wav),
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "pcm_f32le",
                "-shortest", "-y", str(final_output_video)
            ]

            run_command_with_progress(
                cmd_remux, description="Final Remux",
                total_duration=video_dur
            )

        else:
            # === STANDARD HYBRID PIPELINE ===
            # Define Paths
            separation_out_dir = work_dir / "separation"

            # 1. Extract High-Res Audio
            _extract_audio_step(video_path, original_wav, total_duration=video_dur)

            # 2. Split with BSRoformer (Single Pass)
            # Yields Vocals and Background (Instrumental)
            vocals_wav, background_wav = _separate_stems_step(
                original_wav, separation_out_dir, total_duration=video_dur
            )

            # Prepare Directories
            enhanced_vocals_dir = work_dir / "enhanced_vocals"
            denoised_background_dir = work_dir / "denoised_background"

            # 3 & 4. Enhance Vocals & Denoise Background
            # Note: Parallel execution disabled to prevent progress bar contention/interleaving
            # on stdout. The 5090 is fast enough that sequential is preferred for clean UI.

            # if GPU_VRAM_GB >= 22:
            #     log_msg("  [System] ULTRA Profile: Parallel Enhancement/Denoising...")
            #     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #         f_enhance = executor.submit(
            #             _enhance_vocals_step,
            #             vocals_wav, enhanced_vocals_dir, work_dir, total_duration=video_dur
            #         )
            #         f_denoise = executor.submit(
            #             _denoise_background_step,
            #             background_wav, denoised_background_dir, total_duration=video_dur
            #         )
            #
            #         enhanced_vocals_wav = f_enhance.result()
            #         denoised_background_wav = f_denoise.result()
            # else:
            # Sequential Processing (Always for stability)

            enhanced_vocals_wav = _enhance_vocals_step(
                vocals_wav, enhanced_vocals_dir, work_dir, total_duration=video_dur
            )
            denoised_background_wav = _denoise_background_step(
                background_wav, denoised_background_dir, total_duration=video_dur
            )

            # 5. Smart Audio Sync (Cross-Correlation / DTW)
            log_msg("  [Step 4/5] Smart Audio Sync (Sequential for clean output)...")
            # NOTE: We run SEQUENTIALLY now to prevent garbled progress bar output.
            # The progress bar is more valuable to the user than saving ~30s on a 10m task.
            log_msg("    [Info] Syncing Stems (Sequential for clean output)...")

            aligned_vocals = work_dir / f"aligned_{enhanced_vocals_wav.name}"
            _align_stems(original_wav, enhanced_vocals_wav, aligned_vocals)

            aligned_background = work_dir / f"aligned_{denoised_background_wav.name}"
            _align_stems(original_wav, denoised_background_wav, aligned_background)

            # 6. Final Mix
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
        # Cleanup Temp Directory only on absolute success to allow debugging
        if work_dir.exists() and not KEEP_INPUT_FILES:
            if is_valid_video(final_output_video):
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass
            else:
                log_msg(f"  [System] Preservation: Keeping {work_dir.name} for inspection on failure.", level="DEBUG")


def _show_banner(cpu_name, gpu_name):
    """Prints the application banner and hardware info."""
    print("=" * 60)
    print("   AI HYBRID VHS AUDIO RESTORER - v1.0.0")  # pragma: no cover
    print(f"   Running on: {platform.system()} {platform.release()}")  # pragma: no cover
    print("=" * 60 + "\n")  # pragma: no cover

    print("[HARDWARE DETECTED]")
    print(f"   CPU : {os.cpu_count()} Logical Cores ({cpu_name})")
    print(f"   GPU : {gpu_name} ({GPU_VRAM_GB:.2f} GB VRAM)\n")

    print(f"[AUTO-TUNED SETTINGS -> Profile: {PROFILE_NAME}]")
    print("   Audio Precision : 32-bit Float (WAV)")
    print(f"   Process Mode    : {PROCESS_MODE.replace('_', ' ').title()}")
    print(f"   Batch Size      : {GPU_BATCH_SIZE}")
    print(f"   Threads         : {CPU_THREADS}")
    print(f"   Mix Levels      : Vocals={VOCAL_MIX_VOL}, Background={BACKGROUND_MIX_VOL}")
    print(f"   Models          : {VOCALS_MODEL} / UVR-DeNoise")
    print(f"   Config Source   : {CONFIG_SOURCE}\n")


def _scan_files_in_path(path):
    """Scans a file or directory for valid video files."""
    files = []
    if path.is_file():
        if path.suffix.lower() in EXTS:
            files.append(path)
        else:
            print(f">> [Error] Unsupported extension: {path.suffix}")
            print(f">> Supported: {EXTS}")
    elif path.is_dir():
        print(f">> Scanning folder: {path.name}")
        files = [
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in EXTS
            and "_Hybrid_Cleaned" not in f.name
        ]
    else:
        print(f">> [Error] Path is not a file or directory: {path}")

    return files


def _clean_user_input(user_input):
    """Cleans and normalizes user input path."""
    user_input = user_input.strip()

    if user_input.startswith("&"):
        user_input = user_input[1:].strip()

    if user_input.lower().startswith("file://"):
        user_input = user_input[7:].strip()

    if user_input.startswith('"') and user_input.endswith('"'):
        user_input = user_input[1:-1]
    elif user_input.startswith("'") and user_input.endswith("'"):
        user_input = user_input[1:-1]

    return user_input


def _get_interactive_files():
    """Prompts user for input and scans."""
    try:
        print(">> Please Drag & Drop a video file here and press Enter:")
        user_input = input(">>Path: ")
        clean_input = _clean_user_input(user_input)

        if not clean_input:
            print(">> Interactive Mode: Drag & Drop files or press "
                  "Enter to scan 'input' folder.")
            print(">> Scanning 'input' folder...")
            INPUT_DIR.mkdir(exist_ok=True)
            files = [
                f for f in INPUT_DIR.iterdir()
                if f.suffix.lower() in EXTS
                and "_Hybrid_Cleaned" not in f.name
            ]
            return files, False  # use_source_as_output = False

        path = Path(clean_input)
        if not path.exists():
            print(f">> [Error] File not found: {path}")
            return [], False

        files = _scan_files_in_path(path)
        return files, True

    except (EOFError, KeyboardInterrupt):
        return [], False


def _get_input_files():

    files = []
    use_source_as_output = False

    if len(sys.argv) > 1:
        use_source_as_output = True
        print(f">> Arguments Detected: {len(sys.argv) - 1} items")
        for arg in sys.argv[1:]:
            path = Path(arg)
            # Re-use the scanning logic?
            # Original logic handled iterdir slightly different but _scan_files_in_path is robust.
            found = _scan_files_in_path(path)
            files.extend(found)
    else:
        files, use_source_as_output = _get_interactive_files()

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
    OUTPUT_DIR.mkdir(exist_ok=True)
    # TEMP_DIR.mkdir(exist_ok=True) <- Removed

    files, use_source_as_output = _get_input_files()

    if not files:
        print(">> No valid video files found.")
        return

    print(f"\n[System] Found {len(files)} files in queue.")
    print("[System] Starting Batch Processing...")

    for video_path in files:
        # User requested output to ALWAYS be in the input folder
        process_hybrid_audio(
            video_path, gpu_name, target_output_dir=video_path.parent
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
