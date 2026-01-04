
import os
from pathlib import Path
import subprocess
import time


def test_pipeline():
    # Setup
    base_dir = Path(os.getcwd())
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    log_file = base_dir / "session_log.txt"

    # Clean previous run artifacts
    if output_dir.exists():
        for f in output_dir.glob("*_Hybrid_Cleaned*"):
            f.unlink()
    if log_file.exists():
        log_file.unlink()

    # Find any video file in input
    exts = {'.mp4', '.mkv', '.avi', '.mov'}
    video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in exts]

    if not video_files:
        print(f"FATAL: No video files found in {input_dir}")
        return

    input_video = video_files[0]
    print(f"Using test video: {input_video.name}")

    print("=== STARTING E2E TEST ===")
    start_time = time.time()

    # Run the script
    cmd = ["python", "restore_audio_hybrid.py"]

    # Inject Test Mode Env Var
    test_env = os.environ.copy()
    test_env["AI_RESTORE_TEST_MODE"] = "1"

    try:
        subprocess.run(cmd, check=True, env=test_env)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with code {e.returncode}")
        # Note: The script might be designed to catch exceptions and log them,
        # so a 0 exit code
        # doesn't guarantee success, we must check output.

    end_time = time.time()
    print(f"=== EXECUTION FINISHED in {end_time - start_time:.2f}s ===")

    # Verify Output
    expected_output = output_dir / \
        f"{input_video.stem}_Hybrid_Cleaned{input_video.suffix}"
    if expected_output.exists():
        print(f"SUCCESS: Output file generated: {expected_output.name}")
        print(f"Size: {expected_output.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(f"FAILURE: Output file NOT found: {expected_output.name}")

    # Verify Log content
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as log_f:
            logs = log_f.read()

        print("\n--- LOG ANALYSIS ---")
        if "Attempting execution on GPU" in logs:
            print("GPU Usage Attempt: DETECTED (Good)")
        else:
            print("GPU Usage Attempt: NOT DETECTED (Bad)")

        if "GPU Failed" in logs:
            print("GPU Status: FAILED (Fallback to CPU used)")
        else:
            print("GPU Status: SUCCESS (Presumably)")

    else:
        print("FAILURE: Log file not found.")


if __name__ == "__main__":
    test_pipeline()
