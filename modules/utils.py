import torch
import os
import subprocess
import shutil
import sys
import datetime
from pathlib import Path
import re
import threading
import time
import collections
import soundfile as sf
import atexit
import signal
from typing import Set

# Import config constants
from .config import LOG_FILE, DEBUG_LOGGING, EXTS


# === AUTO-CONFIGURE PATH ===
project_dir = Path(__file__).parent.parent.resolve()  # modules/..
venv_scripts = project_dir / "venv" / "Scripts"

FFMPEG_BIN = "ffmpeg"
if (venv_scripts / "ffmpeg.exe").exists():
    FFMPEG_BIN = str(venv_scripts / "ffmpeg.exe")
    current_path = os.environ.get("PATH", "")
    if str(venv_scripts) not in current_path:
        os.environ["PATH"] = str(venv_scripts) + os.pathsep + current_path
elif venv_scripts.exists():
    current_path = os.environ.get("PATH", "")
    if str(venv_scripts) not in current_path:
        os.environ["PATH"] = str(venv_scripts) + os.pathsep + current_path

_venv_scripts_missing = not venv_scripts.exists()


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


if _venv_scripts_missing:
    log_msg(f"Venv Scripts not found at: {project_dir / 'venv' / 'Scripts'}", level="DEBUG")


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
        except Exception:
            pass

    # Give them a moment to terminate gracefully
    time.sleep(0.5)

    for p in list(_active_processes):
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
    _active_processes.clear()


def _len_active():
    return len(_active_processes)


def signal_handler(sig, frame):
    """Handles termination signals."""
    log_msg("\n[System] Termination signal received. Stopping...", is_error=True)
    cleanup_subprocesses()
    sys.exit(1)


# Register handlers
atexit.register(cleanup_subprocesses)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if sys.platform == "Windows":
    sig_break = getattr(signal, "SIGBREAK", None)
    if sig_break is not None:
        signal.signal(sig_break, signal_handler)


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


def format_time(seconds):
    """Formats seconds as HH:MM:SS.mm"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


def parse_ffmpeg_time(line):
    """Extracts time=HH:MM:SS.mm from FFmpeg output."""
    match = re.search(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})", line)
    if match:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 100.0
    return None

# Import drawing from UI - WAIT, UI depends on Utils for time/etc.
# Circular dependency risk if UI needs format_time and Utils needs draw_progress_bar.
# run_command_with_progress calls draw_progress_bar.
# So utils needs ui?
# Let's put UI components (draw_progress_bar) in utils OR
# move run_command_with_progress to ui?
# run_command_with_progress is logic, but has UI.
# Maybe I should put draw_progress_bar in utils as well?
# Or make a separate `ui_base`?
# I will put draw_progress_bar and format_time ALL in utils.py for now to avoid circular deps.
# The plan said `modules/ui.py` has `draw_progress_bar`.
# If I stick to the plan, `utils` needs to import `ui`. `ui` needs `utils` (for format_time?).
# Logic: `utils` is low level. `ui` is high level.
# `run_command_with_progress` is high level util?
# I'll put `draw_progress_bar` and `run_command_with_progress` ALL in `utils.py` for simplicity/robustness against circular deps.
# And `modules/ui.py` will just have the CLI/Banner stuff (`_show_banner`, `_get_input_files`).
# That seems safer.


# Global Thread Locks and State for UI
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


def _get_terminal_columns(default=79):
    """Returns terminal columns with safety margin."""
    try:
        # Remove 80-column cap for modern terminals
        return shutil.get_terminal_size((80, 20)).columns - 1
    except Exception:
        return default


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


# Need to import torch at end or lazily?
# attempt_run_with_retry uses torch.cuda.is_available.
# I need 'import torch' at top.


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
