from unittest.mock import MagicMock, patch
import subprocess
from pathlib import Path
import pytest
import modules.utils

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------


def test_is_valid_audio_robust(tmp_path):
    """Test audio validation."""
    p = tmp_path / "v.wav"
    p.write_text("x" * 1500)
    with patch("modules.utils.sf.SoundFile") as mock_sf:
        mock_ctx = mock_sf.return_value.__enter__.return_value
        mock_ctx.frames = 50000  # > 0.1s
        mock_ctx.samplerate = 44100
        assert modules.utils.is_valid_audio(p) is True
    # Test internal check for non-existence logic is covered by is_valid_audio implementation
    assert modules.utils.is_valid_audio(tmp_path / "no") is False


def test_is_valid_audio_small_file(tmp_path):
    """Test is_valid_audio with small files."""
    # File smaller than 1KB
    small_file = tmp_path / "small.wav"
    small_file.write_text("x" * 500)
    assert modules.utils.is_valid_audio(small_file) is False


@patch("modules.utils.sf.SoundFile")
def test_is_valid_audio_zero_frames(mock_sf, tmp_path):
    """Test is_valid_audio with zero frames."""
    p = tmp_path / "empty.wav"
    p.write_text("x" * 1500)
    mock_sf.return_value.__enter__.return_value.frames = 0
    assert modules.utils.is_valid_audio(p) is False


@patch("modules.utils.sf.SoundFile")
def test_is_valid_audio_exception(mock_sf, tmp_path):
    """Test is_valid_audio handles SoundFile exceptions."""
    p = tmp_path / "corrupt.wav"
    p.write_text("x" * 2000)
    mock_sf.side_effect = Exception("Corrupt file")
    assert modules.utils.is_valid_audio(p) is False


def test_is_valid_video_small(tmp_path):
    """Test is_valid_video rejects files smaller than 1MB."""
    p = tmp_path / "small.mp4"
    p.write_text("x" * 500)  # 500 bytes
    assert modules.utils.is_valid_video(p) is False


def test_retry_loop():
    """Test retry loop with batch size reduction."""
    # First call fails, second succeeds
    mock_proc_fail = MagicMock()
    mock_proc_fail.wait.return_value = None
    mock_proc_fail.returncode = 1  # Failure

    mock_proc_success = MagicMock()
    mock_proc_success.wait.return_value = None
    mock_proc_success.returncode = 0  # Success

    # Fix: Mock stdout.readline to avoid regex TypeError
    mock_proc_fail.stdout.readline.return_value = ""
    mock_proc_success.stdout.readline.return_value = ""

    with patch("modules.utils.subprocess.Popen", side_effect=[mock_proc_fail, mock_proc_success]):
        # Test with batch size > 1 so it can retry
        result = modules.utils.attempt_run_with_retry(lambda b: ["echo", str(b)], 2)
        assert result is True


@patch("modules.utils.subprocess.run")
def test_deps_fail(mock_run):
    """Test dependency check failure."""
    # Use FileNotFoundError since that's what check_dependencies catches
    mock_run.side_effect = FileNotFoundError("ffmpeg not found")
    assert modules.utils.check_dependencies() is False


def test_draw_progress_bar(capsys):
    """Test progress bar rendering."""
    # Reset global_state to ensure bar is drawn
    modules.utils._last_bar_time = 0
    modules.utils.draw_progress_bar(50, "Testing...")
    captured = capsys.readouterr()
    assert "50.0%" in captured.out
    assert "Testing..." in captured.out

    modules.utils.draw_progress_bar(0, "Start")
    captured = capsys.readouterr()
    assert "0.0%" in captured.out

    modules.utils.draw_progress_bar(100, "Done")
    captured = capsys.readouterr()
    assert "100.0%" in captured.out

    modules.utils.draw_progress_bar(150, "Over")
    captured = capsys.readouterr()
    assert "100.0%" in captured.out

    modules.utils.draw_progress_bar(-10, "Under")
    captured = capsys.readouterr()
    assert "0.0%" in captured.out


def test_log_msg_variations(tmp_path, capsys):
    """Test log message variations."""
    # Temporarily change LOG_FILE
    original_log = modules.utils.LOG_FILE
    modules.utils.LOG_FILE = tmp_path / "test_log.txt"

    try:
        # Normal message
        modules.utils.log_msg("Test message", console=True)
        captured = capsys.readouterr()
        assert "Test message" in captured.out

        # Error message
        modules.utils.log_msg("Error!", is_error=True)
        captured = capsys.readouterr()
        assert "Error!" in captured.out

        # Debug message (should NOT print to console)
        modules.utils.log_msg("Debug info", level="DEBUG")
        captured = capsys.readouterr()
        assert "Debug info" not in captured.out

        # Console=False
        modules.utils.log_msg("Silent", console=False)
        captured = capsys.readouterr()
        assert "Silent" not in captured.out

        # Verify log file was written
        log_content = modules.utils.LOG_FILE.read_text()
        assert "Test message" in log_content
        assert "ERROR" in log_content  # Error should be logged as ERROR

    finally:
        modules.utils.LOG_FILE = original_log


def test_log_msg_file_error(tmp_path, capsys, monkeypatch):
    """Test log_msg handles file write errors gracefully."""
    # Set LOG_FILE to an invalid path
    original = modules.utils.LOG_FILE
    modules.utils.LOG_FILE = Path("/nonexistent/path/log.txt")

    try:
        # Should not raise, just silently fail file write
        modules.utils.log_msg("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out
    finally:
        modules.utils.LOG_FILE = original


def test_parse_ffmpeg_time():
    """Test FFmpeg time parsing."""
    # Standard format
    result = modules.utils.parse_ffmpeg_time("time=01:23:45.67")
    assert result == pytest.approx(1 * 3600 + 23 * 60 + 45 + 0.67, rel=0.01)

    # Zero time
    result = modules.utils.parse_ffmpeg_time("time=00:00:00.00")
    assert result == 0.0

    # No match
    result = modules.utils.parse_ffmpeg_time("some random text")
    assert result is None


def test_format_time_negative():
    """Test format_time with negative input."""
    assert modules.utils.format_time(-10) == "00:00:00,000"


@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_passthrough(mock_popen, capsys):
    """Test run_command_with_progress without duration (passthrough mode)."""
    mock_proc = MagicMock()
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0
    # Fix: Mock stdout.readline
    mock_proc.stdout.readline.return_value = ""
    mock_popen.return_value = mock_proc

    # Run without total_duration - passthrough mode
    modules.utils.run_command_with_progress(
        ["echo", "test"],
        description="Testing passthrough"
    )
    captured = capsys.readouterr()
    assert "Testing passthrough" in captured.out


@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_passthrough_fail(mock_popen):
    """Test run_command_with_progress failure in passthrough mode."""
    mock_proc = MagicMock()
    mock_proc.wait.return_value = None
    mock_proc.returncode = 1  # Failure
    # Fix: Mock stdout.readline
    mock_proc.stdout.readline.return_value = ""
    mock_popen.return_value = mock_proc

    with pytest.raises(subprocess.CalledProcessError):
        modules.utils.run_command_with_progress(["fail_cmd"])


@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_ffmpeg(mock_popen):
    """Test run_command_with_progress with FFmpeg progress parsing."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    # Fix: Mock stdout.readline to avoid regex TypeError
    mock_proc.stdout.readline.return_value = ""

    # Create a list of lines to return, then empty forever
    lines = [
        "frame=100 time=00:00:05.00 bitrate=1000kbits/s",
        "frame=200 time=00:00:10.00 bitrate=1000kbits/s",
        "",  # Empty signals end
    ]
    line_iter = iter(lines)

    def readline_mock():
        try:
            return next(line_iter)
        except StopIteration:
            return ""

    mock_proc.stderr.readline = readline_mock

    # poll returns None while there's output, then 0 when done
    poll_values = [None, None, 0]
    poll_iter = iter(poll_values)

    def poll_mock():
        try:
            return next(poll_iter)
        except StopIteration:
            return 0

    mock_proc.poll = poll_mock
    mock_popen.return_value = mock_proc

    modules.utils.run_command_with_progress(
        ["ffmpeg", "-i", "input.mp4", "output.mp4"],
        total_duration=20.0,
        description="Encoding"
    )
    mock_proc.wait.assert_called()


@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_ffmpeg_fail(mock_popen):
    """Test run_command_with_progress FFmpeg failure."""
    mock_proc = MagicMock()
    mock_proc.returncode = 1  # Failure
    mock_proc.poll.return_value = 1
    # Fix: Mock stdout.readline to avoid regex TypeError
    mock_proc.stdout.readline.return_value = ""
    mock_proc.stderr.readline.return_value = ""
    mock_popen.return_value = mock_proc

    with pytest.raises(subprocess.CalledProcessError):
        modules.utils.run_command_with_progress(
            ["ffmpeg"],
            total_duration=10.0
        )


@patch("modules.utils.run_command_with_progress")
def test_attempt_cpu_run_with_retry_success(mock_run):
    """Test CPU retry on first success."""
    mock_run.return_value = None  # Success
    result = modules.utils.attempt_cpu_run_with_retry(
        lambda t: ["cmd", str(t)],
        initial_threads=8,
        description="CPU Task"
    )
    assert result is True
    mock_run.assert_called_once()


@patch("modules.utils.run_command_with_progress")
def test_attempt_cpu_run_with_retry_fallback(mock_run):
    """Test CPU retry with thread reduction."""
    # First call fails, second succeeds
    mock_run.side_effect = [
        subprocess.CalledProcessError(1, "cmd"),
        None  # Success
    ]

    result = modules.utils.attempt_cpu_run_with_retry(
        lambda t: ["cmd", str(t)],
        initial_threads=4,
        description="CPU Fallback"
    )
    assert result is True
    assert mock_run.call_count == 2


@patch("modules.utils.run_command_with_progress")
def test_attempt_cpu_run_with_retry_exhausted(mock_run):
    """Test CPU retry when all threads exhausted."""
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")

    with pytest.raises(subprocess.CalledProcessError):
        modules.utils.attempt_cpu_run_with_retry(
            lambda t: ["cmd", str(t)],
            initial_threads=1
        )


def test_attempt_run_with_retry_error_path():
    """Test attempt_run_with_retry error logging branch."""
    with patch("modules.utils.subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = ""
        mock_proc.wait.return_value = None
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        # Should trigger warning log on first failure with batch > 1
        with patch("modules.utils.log_msg") as mock_log:
            with pytest.raises(subprocess.CalledProcessError):
                modules.utils.attempt_run_with_retry(lambda b: ["cmd"], 1)

            # Reset and try with batch > 1
            mock_popen.side_effect = [mock_proc, mock_proc]  # Fail twice
            with pytest.raises(subprocess.CalledProcessError):
                modules.utils.attempt_run_with_retry(lambda b: ["cmd"], 2)
            assert mock_log.called


def test_cleanup_subprocesses_none():
    """Test cleanup when no processes are active."""
    modules.utils._active_processes.clear()
    modules.utils.cleanup_subprocesses()
    # Should just return without error


def test_cleanup_subprocesses_active():
    """Test cleanup with active processes."""
    mock_p1 = MagicMock(spec=subprocess.Popen)
    mock_p1.poll.return_value = None

    mock_p2 = MagicMock(spec=subprocess.Popen)
    mock_p2.poll.return_value = 0  # Already done

    modules.utils._active_processes.add(mock_p1)
    modules.utils._active_processes.add(mock_p2)

    # Mock time.sleep to speed up test
    with patch("time.sleep"):
        modules.utils.cleanup_subprocesses()

    mock_p1.terminate.assert_called()
    mock_p1.kill.assert_called()
    assert len(modules.utils._active_processes) == 0


def test_len_active():
    """Test _len_active utility."""
    modules.utils._active_processes.clear()
    assert modules.utils._len_active() == 0
    modules.utils._active_processes.add(MagicMock())
    assert modules.utils._len_active() == 1


@patch("modules.utils.draw_progress_bar")
@patch("modules.utils.subprocess.Popen")
def test_run_command_with_progress_adds_to_active(mock_popen, mock_bar):
    """Verify process is added to _active_processes during execution."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    # Return empty string immediately to finish loop
    mock_proc.stdout.readline.return_value = ""
    mock_popen.return_value = mock_proc
    
    # We want to check _active_processes while it's running.
    # We can do this by making wait() check the set.
    def wait_side_effect():
        assert len(modules.utils._active_processes) >= 1
        return 0
    
    mock_proc.wait.side_effect = wait_side_effect

    modules.utils.run_command_with_progress(["cmd"])
    
    # After it finishes, it should be removed (checked by other tests probably)
    # But inside wait(), it was active.
    assert mock_proc.wait.called



def test_check_dependencies_success():
    """Test check_dependencies returns True when all found."""
    with patch("subprocess.run") as mock_run:
        assert modules.utils.check_dependencies() is True


def test_cleanup_subprocesses_exception():
    """Test cleanup handling of strict exceptions."""
    p1 = MagicMock()
    p1.poll.return_value = None
    p1.terminate.side_effect = Exception("Fail")
    p1.kill.side_effect = Exception("Fail")
    
    with patch("modules.utils._active_processes", {p1}):
        modules.utils.cleanup_subprocesses()
        # Should not raise
        assert p1.terminate.called


def test_adjust_layout_extreme_truncate():
    """Test layout strategy 3: truncation."""
    # Force very narrow columns
    width, info, label = modules.utils._adjust_bar_layout(
        width=20, info_str="INFO", label="VeryLongLabelThatNeedsTruncation", columns=30
    )
    assert "..." in label
    assert len(label) < 20


@patch("modules.utils.sf.write")
@patch("modules.utils.is_valid_audio", return_value=False)
def test_save_atomic_fail_cleanup(mock_valid, mock_write, tmp_path):
    """Test atomic save cleans up invalid temp file."""
    f = tmp_path / "test.wav"
    modules.utils._save_audio_atomic(f, [], 44100)
    # Temp file should be unlinked
    assert not (f.with_suffix(".tmp.wav")).exists()


@patch("modules.utils.draw_progress_bar")
@patch("modules.utils.subprocess.Popen")
def test_monitor_progress_tqdm(mock_popen, mock_draw):
    """Test TQDM progress parsing in run_command_with_progress."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    # Ensure has no _last_pc initially (or set it to -1)
    # properly mocking hasattr is hard, so let's set it to valid float start
    mock_proc._last_pc = -1.0

    mock_proc.poll.side_effect = [None, None, 0]  # Run loop twice then exit

    # TQDM output simulation
    lines = [
        "45%|xxxx| 10/20 [00:10<00:10, 1.00it/s]",
        "100%|xxxx| 20/20 [00:20<00:00, 1.00it/s]",
        ""
    ]
    # Robust iteration for readline
    mock_proc.stdout.readline.side_effect = lines + [""] * 5
    mock_popen.return_value = mock_proc

    modules.utils.run_command_with_progress(["cmd"], description="TQDM Test")

    # Verify 45% was drawn
    # Note: draw_progress_bar args: (percent, label, ...)
    calls = mock_draw.call_args_list
    found_45 = any(c[0][0] == 45.0 for c in calls)
    assert found_45, "Did not find 45% progress update"


@patch("modules.utils.draw_progress_bar")
@patch("modules.utils.subprocess.Popen")
def test_monitor_progress_ffmpeg_time(mock_popen, mock_draw):
    """Test FFmpeg 'time=' parsing."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    # Provide enough poll side_effects
    mock_proc.poll.side_effect = [None, None, 0]

    # FFmpeg time output (15s)
    lines = [
        "frame=100 time=00:00:15.00 bitrate=...",
        ""
    ]
    # Robust iteration
    mock_proc.stdout.readline.side_effect = lines + [""] * 5
    mock_popen.return_value = mock_proc

    # Total duration 30s -> 15s should be 50%
    modules.utils.run_command_with_progress(
        ["ffmpeg"],
        total_duration=30.0,
        description="FFmpeg Test"
    )

    calls = mock_draw.call_args_list
    # Look for approx 50%
    found_50 = any(abs(c[0][0] - 50.0) < 0.1 for c in calls)
    assert found_50, "Did not find expected ~50% progress update"
