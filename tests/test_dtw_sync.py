
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path
import modules.sync


def test_parallel_dtw_logic():
    """Test chunking and stitching logic."""

    # 1. Create simulated features (10k frames)
    N = 10000
    # 1. Create simulated features (10k frames)
    N = 10000

    # Patch concurrent.futures.as_completed to return futures immediately
    # This prevents the test from hanging when using Mock objects as futures
    with patch("concurrent.futures.ProcessPoolExecutor") as MockExecutor, \
            patch("concurrent.futures.as_completed") as mock_as_completed:

        mock_pool = MagicMock()
        MockExecutor.return_value = mock_pool
        mock_pool.__enter__.return_value = mock_pool

        # Setup as_completed to return list of futures immediately
        mock_as_completed.side_effect = lambda futures: list(futures)

        # Mock submit to return a mock future with a result
        def side_effect_submit(func, chunk):
            mock_future = MagicMock()
            ref_seg, proc_seg, rad = chunk
            # Return diagonal path for this chunk size
            L = len(ref_seg)
            path = [[i, i] for i in range(L)]
            mock_future.result.return_value = path
            return mock_future

        mock_pool.submit.side_effect = side_effect_submit

        # We need to mock librosa load/chroma too to reach the logic
        with patch("modules.sync.CPU_THREADS", 4), \
                patch("modules.sync.DTW_RESOLUTION", 40), \
                patch("modules.sync.GPU_VRAM_GB", 0):

            # Configure the mock that will be injected
            mock_librosa = MagicMock()
            mock_librosa.load.return_value = (np.zeros(N * 100), 8192)  # dummy audio
            # features (12 x N)
            mock_librosa.feature.chroma_stft.return_value = np.zeros((12, N))

            ref = Path("ref.wav")
            proc = Path("proc.wav")
            out = Path("out.wav")

            # Call
            # We need to bypass the "import librosa" check
            modules.sync.librosa = mock_librosa
            modules.sync.fastdtw = MagicMock()

            # But _align_stems_dtw is what we test
            # It calls sf.read/write at the end. Patch them.
            with patch("modules.sync.sf.read", return_value=(np.zeros((N * 100, 2)), 44100)), \
                    patch("modules.sync.sf.write"):

                modules.sync._align_stems_dtw(ref, proc, out)

                # Verify parallelism
                # It uses submit now, not map
                assert mock_pool.submit.call_count >= 4

                # Verify chunks were created
                args, _ = mock_pool.submit.call_args_list[0]
                chunk_arg = args[1]
                assert len(chunk_arg[0]) == 3000


def test_gpu_dtw_worker():
    """Test GPU DTW worker logic with mocks."""

    # Creates 10 frames, 12 features
    ref_seg = np.zeros((10, 12))
    proc_seg = np.zeros((10, 12))
    radius = 5

    # Mock torch and librosa
    with patch.dict("sys.modules", {"torch": MagicMock(), "librosa": MagicMock()}):
        import torch
        import librosa

        # Setup Torch mocks
        torch.cuda.is_available.return_value = True
        mock_tensor = MagicMock()
        torch.tensor.return_value = mock_tensor

        # cdist returns a tensor, which we call .cpu().numpy() on
        mock_dist_tensor = MagicMock()
        mock_dist_matrix = np.zeros((10, 10))  # 10x10 cost matrix
        mock_dist_tensor.cpu.return_value.numpy.return_value = mock_dist_matrix
        torch.cdist.return_value = mock_dist_tensor

        # Setup Librosa mock
        mock_path = np.array([[9, 9], [0, 0]])  # End to Start
        librosa.sequence.dtw.return_value = (None, mock_path)

        # Call the worker directly
        args = (ref_seg, proc_seg, radius)
        result_path = modules.sync._run_gpu_dtw_chunk(args)

        # Assertions
        assert torch.cdist.called
        assert librosa.sequence.dtw.called

        # Result should be REVERSED path (Start to End)
        assert result_path[0] == [0, 0]
        assert result_path[1] == [9, 9]


def test_align_stems_dtw_dispatches_gpu():
    """Test that _align_stems_dtw choses GPU worker/executor when hardware allows."""
    ref = Path("ref.wav")
    proc = Path("proc.wav")
    out = Path("out.wav")

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    # Use patch.dict for torch
    with patch.dict("sys.modules", {"torch": mock_torch}):
        with patch("modules.sync.GPU_VRAM_GB", 10), \
                patch("concurrent.futures.ThreadPoolExecutor") as MockThreadPool, \
                patch("concurrent.futures.ProcessPoolExecutor") as MockProcessPool, \
                patch("modules.sync.librosa") as mock_librosa, \
                patch("modules.sync.sf.read", return_value=(np.zeros((1000, 2)), 44100)), \
                patch("modules.sync.sf.write"):

            # Setup librosa to return dummy features
            mock_librosa.load.return_value = (np.zeros(100), 8192)
            mock_librosa.feature.chroma_stft.return_value = np.zeros((12, 10))

            # We also need to mock as_completed here since it's used even for GPU path (via ThreadPool)
            with patch("concurrent.futures.as_completed", side_effect=lambda futures: list(futures)):
                modules.sync._align_stems_dtw(ref, proc, out)

            # Should verify that ThreadPool was used (for GPU), NOT ProcessPool
            assert MockThreadPool.called
            assert not MockProcessPool.called
