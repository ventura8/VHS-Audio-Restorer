import sys
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path
import modules.sync

# ---------------------------------------------------------
# Sync Logic
# ---------------------------------------------------------


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
def test_align_stems_branches(mw, mr):
    """Test audio alignment branches."""
    sr = 44100
    mr.return_value = (np.zeros((100, 2)), sr)

    # Negligible lag
    with patch("scipy.signal.correlation_lags", return_value=np.array([0])):
        with patch("modules.sync.SYNC_METHOD", "shift"):
            modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))
            assert mw.called

    # Positive lag
    with patch("scipy.signal.correlation_lags", return_value=np.array([50])):
        with patch("modules.sync.SYNC_METHOD", "shift"):
            modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))

    # Negative lag
    with patch("scipy.signal.correlation_lags", return_value=np.array([-50])):
        with patch("modules.sync.SYNC_METHOD", "shift"):
            modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
def test_align_stems_mono_to_stereo(mock_write, mock_read):
    """Test align_stems converts mono to stereo."""
    sr = 44100
    # Return mono audio
    mock_read.return_value = (np.zeros((100, 1)), sr)

    with patch("modules.sync.SYNC_METHOD", "shift"):
        modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))

    # Verify stereo output was written
    call_args = mock_write.call_args
    written_data = call_args[0][1]  # Second positional arg is the data
    assert written_data.shape[1] == 2  # Should be stereo


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
def test_align_stems_empty_audio(mock_write, mock_read):
    """Test align_stems handles empty audio."""
    sr = 44100
    # Return empty audio
    mock_read.return_value = (np.zeros((0, 2)), sr)

    with patch("modules.sync.SYNC_METHOD", "shift"):
        modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))
    mock_write.assert_called()


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
def test_align_stems_exception_fallback(mock_write, mock_read):
    """Test align_stems fallback on exception."""
    mock_read.side_effect = Exception("Read error")

    with patch("modules.sync.SYNC_METHOD", "shift"):
        modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))
    # Should fall back to write raw which might involve read.
    # If read fails again inside fallback, it returns.
    pass


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
def test_align_stems_large_negative_lag(mock_write, mock_read):
    """Test align_stems with very large negative lag."""
    sr = 44100
    # Return small audio
    mock_read.return_value = (np.zeros((50, 2)), sr)

    # Simulate very large negative lag (cut more than available)
    with patch("scipy.signal.correlation_lags", return_value=np.array([-1000])):
        modules.sync._align_stems(Path("o.wav"), Path("p.wav"), Path("out.wav"))
        mock_write.assert_called()


def test_apply_warp_gpu_mocked():
    """Test _apply_warp_gpu with mocked PyTorch."""
    mock_torch = MagicMock()
    # Setup mocks
    mock_torch.cuda.is_available.return_value = True
    mock_device = MagicMock()
    mock_torch.device.return_value = mock_device

    # Mock tensor creation and methods
    mock_tensor = MagicMock()
    mock_torch.from_numpy.return_value = mock_tensor
    mock_tensor.float.return_value = mock_tensor
    mock_tensor.unsqueeze.return_value = mock_tensor
    mock_tensor.to.return_value = mock_tensor

    # Mock grid_sample result
    mock_warped = MagicMock()
    mock_torch.nn.functional.grid_sample.return_value = mock_warped

    # Mock chain: squeeze -> permute -> cpu -> numpy
    mock_cpu = MagicMock()
    mock_warped.squeeze.return_value.squeeze.return_value.permute.return_value.cpu.return_value = mock_cpu
    expected_output = np.array([[0.1, 0.2]])
    mock_cpu.numpy.return_value = expected_output

    # Input data
    audio_np = np.zeros((10, 2))
    indices_np = np.zeros(10)

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = modules.sync._apply_warp_gpu(audio_np, indices_np)

        assert result is expected_output
        mock_torch.nn.functional.grid_sample.assert_called_once()


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
def test_align_stems_shift_empty_mono(mock_write, mock_read, tmp_path):
    """Test clean fallback for empty audio in align steps."""
    # First read (ref/proc check) -> returns empty arrays
    # Second read (fallback write) -> returns mono array

    mock_read.side_effect = [
        (np.array([]), 44100),  # ref
        (np.array([]), 44100),  # proc
        (np.zeros((100, 1)), 44100)  # fallback read, mono
    ]

    wav = tmp_path / "test.wav"
    out = tmp_path / "out.wav"

    modules.sync._align_stems_shift(wav, wav, out)

    # Verify write was called with stereo data (tiled)
    args = mock_write.call_args
    data = args[0][1]
    assert data.shape[1] == 2  # Should be tiled to stereo


@patch("modules.sync.sf.read")
@patch("modules.sync.sf.write")
def test_align_shift_failure_fallback(mock_write, mock_read, tmp_path, capsys):
    """Test fallback when shift sync fails."""
    wav = tmp_path / "in.wav"
    out = tmp_path / "out.wav"

    # Mock read failure
    # 1. read ref (fail) -> Exception
    # 2. Catch -> read raw (succeed) -> write
    mock_read.side_effect = [Exception("Corrupt Ref"), (np.zeros((10, 2)), 44100)]

    modules.sync._align_stems_shift(wav, wav, out)

    assert mock_write.called
    captured = capsys.readouterr()
    assert "Sync failed" in captured.err or "Sync failed" in captured.out


@patch("modules.sync.map_coordinates")
@patch("modules.sync._save_audio_atomic")
def test_warp_aligned_audio_cpu(mock_save, mock_map, tmp_path):
    """Test CPU warping fallback."""
    # Setup inputs
    proc = np.zeros((100, 2))
    indices = np.zeros(100)
    out = tmp_path / "out.wav"

    mock_map.return_value = np.zeros(100)
    mock_save.return_value = True

    modules.sync._warp_aligned_audio_cpu(proc, indices, 2, out, 44100)

    assert mock_map.call_count == 2  # Once per channel
    assert mock_save.called


def test_run_fastdtw_chunk():
    """Test fastdtw worker function directly."""
    # The worker function imports fastdtw internally.
    # We must patch the usage of the imported module.
    # Since we can't easily patch an import inside a function from outside without
    # complex sys.modules hacks, we'll patch sys.modules dictionary for 'fastdtw'.
    mock_fastdtw_module = MagicMock()
    mock_fastdtw_module.fastdtw.return_value = (0, [(0, 0), (1, 1)])

    with patch.dict("sys.modules", {"fastdtw": mock_fastdtw_module}):
        # fastdtw expects 1D arrays if checking simple distance, or 2D if features.
        # The actual implementation passes features with shape (Frames, 12).
        args = (np.zeros((10, 12)), np.zeros((10, 12)), 10)
        path = modules.sync._run_fastdtw_chunk(args)

    assert path == [(0, 0), (1, 1)]


@patch("modules.sync.librosa", None)
@patch("modules.sync.fastdtw", None)
@patch("modules.sync._align_stems_shift")
def test_align_stems_dtw_missing_deps(mock_shift, tmp_path):
    """Test DTW sync fallback when deps missing."""
    wav = tmp_path / "a.wav"
    out = tmp_path / "out.wav"
    modules.sync._align_stems_dtw(wav, wav, out)
    assert mock_shift.called


def test_apply_warp_gpu_exception(capsys):
    """Test exception handler in GPU warp."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    # Raise exception during processing
    mock_torch.from_numpy.side_effect = Exception("GPU Boom")

    with patch.dict(sys.modules, {"torch": mock_torch}):
        res = modules.sync._apply_warp_gpu(np.zeros((10, 2)), np.zeros(10))
        assert res is None
        captured = capsys.readouterr()
        assert "GPU Warp failed" in captured.out
