#!/usr/bin/env python3
from modules.processing import process_hybrid_audio
from modules.ui import run_init_sequence, _show_banner, _get_input_files
from modules.utils import check_dependencies
from modules.config import OUTPUT_DIR
import sys
import time
from pathlib import Path

# Ensure local modules can be imported
sys.path.append(str(Path(__file__).parent))


def main():
    # 1. Initialization Sequence
    cpu_name, gpu_name = run_init_sequence()

    if not check_dependencies():
        print("\n[Init] Critical Error: Dependencies Missing.")
        return

    # 2. Show Banner
    # We retrieve names again or pass them. _show_banner gets them internally
    # but run_init_sequence returned them.
    # Current UI implementation of _show_banner calls get_cpu_name() again.
    # It prints the banner.
    _show_banner()

    print("-" * 60)
    print(" [HOW TO USE]")
    print(" 1. Drag and Drop a video file (or folder) here.")
    print(" 2. Or paste the file path below.")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 3. Get Inputs
    files, use_source_as_output = _get_input_files()

    if not files:
        print(">> No valid video files found.")
        return

    print(f"\n[System] Found {len(files)} files in queue.")
    print("[System] Starting Batch Processing...")

    # 4. Processing Loop
    for video_path in files:
        # User requested output to ALWAYS be in the input folder (from original logic)
        process_hybrid_audio(
            video_path, gpu_name, target_output_dir=video_path.parent
        )

    print("\n" + "=" * 60)
    print("   BATCH PROCESSING COMPLETE")
    print("=" * 60)
    try:
        input("Press Enter to exit...")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
