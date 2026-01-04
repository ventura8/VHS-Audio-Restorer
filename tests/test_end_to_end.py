import os
from pathlib import Path
import subprocess
import time


def _setup_test_environment(base_dir):
    """Sets up input/output directories and cleans up old artifacts."""
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    log_file = base_dir / "session_log.txt"

    # Clean previous run artifacts
    if output_dir.exists():
        for f in output_dir.glob("*_Hybrid_Cleaned*"):
            f.unlink()
    if log_file.exists():
        log_file.unlink()

    # Ensure directories exist (important for CI)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, output_dir, log_file


def _get_test_video(input_dir):
    """Finds a video file for testing."""
    exts = {'.mp4', '.mkv', '.avi', '.mov'}
    video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in exts]

    if not video_files:
        print(f"SKIPPED: No video files found in {input_dir}. Add a file to run E2E test.")
        return None

    return video_files[0]


def _run_script():
    """Executes the restore script via subprocess."""
    print("=== STARTING E2E TEST ===")
    start_time = time.time()

    cmd = ["python", "restore_audio_hybrid.py"]

    # Inject Test Mode Env Var
    test_env = os.environ.copy()
    test_env["AI_RESTORE_TEST_MODE"] = "1"

    try:
        subprocess.run(cmd, check=True, env=test_env)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with code {e.returncode}")

    end_time = time.time()
    print(f"=== EXECUTION FINISHED in {end_time - start_time:.2f}s ===")


def _verify_output(output_dir, log_file, input_video):
    """Verifies that output files exist and checks logs."""
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


def test_pipeline():
    base_dir = Path(os.getcwd())
    input_dir, output_dir, log_file = _setup_test_environment(base_dir)

    input_video = _get_test_video(input_dir)
    if not input_video:
        return

    print(f"Using test video: {input_video.name}")

    _run_script()
    _verify_output(output_dir, log_file, input_video)


if __name__ == "__main__":
    test_pipeline()
