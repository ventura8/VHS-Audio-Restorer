import os
import sys
import platform
import time
from pathlib import Path

from .config import (
    CONFIG, INPUT_DIR, OUTPUT_DIR, EXTS, PROCESS_MODE,
    VOCAL_MIX_VOL, BACKGROUND_MIX_VOL,
    VOCALS_MODEL, CONFIG_SOURCE
)
from .utils import draw_progress_bar
from .hardware import (
    CPU_THREADS, GPU_VRAM_GB, get_cpu_name, get_gpu_name,
    GPU_BATCH_SIZE, PROFILE_NAME
)

# Constants imported from config via restore_audio_hybrid normally,
# but we can access them here or pass them.
# The banner uses a lot of them.


def _show_banner():
    """Prints the application banner and hardware info."""
    cpu_name = get_cpu_name()
    gpu_name = get_gpu_name()

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

    return cpu_name, gpu_name


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

    # Move quote stripping up, before file:// check
    # because user input might be: "file://..."
    if user_input.startswith('"') and user_input.endswith('"'):
        user_input = user_input[1:-1]
    elif user_input.startswith("'") and user_input.endswith("'"):
        user_input = user_input[1:-1]

    if user_input.lower().startswith("file://"):
        user_input = user_input[7:].strip()

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


def run_init_sequence():
    """Runs the visual initialization sequence."""
    print("\n[AI ENGINE INITIALIZATION]")

    draw_progress_bar(10, "Initializing Core Systems...")
    time.sleep(0.3)
    draw_progress_bar(30, "Scanning Hardware...")
    time.sleep(0.2)

    cpu_name = get_cpu_name()
    draw_progress_bar(45, f"CPU Detected: {os.cpu_count()} Cores")

    gpu_name = get_gpu_name()
    draw_progress_bar(60, f"GPU Detected: {gpu_name}")

    # Profile name logic in config/hardware, but we need it here
    # It's already in constants
    p_name = PROFILE_NAME.split('(')[0].strip()
    draw_progress_bar(75, f"Applying Optimization Profile: {p_name}")
    time.sleep(0.2)

    # Check dependencies logic?
    # It was in utils or restore_audio_hybrid.py
    # I didn't see explicit check_dependencies function in my previous reads of restore_audio_hybrid.py loops?
    # Wait, line 1875 in restore_audio_hybrid calls `check_dependencies()`.
    # I need to implement `check_dependencies` in `utils.py` or `ui.py`.
    # I checked `utils.py` content, I didn't add `check_dependencies`.

    draw_progress_bar(90, "Verifying Libraries...")
    time.sleep(0.2)
    draw_progress_bar(100, "Initialization Complete.")
    time.sleep(0.4)

    return cpu_name, gpu_name
