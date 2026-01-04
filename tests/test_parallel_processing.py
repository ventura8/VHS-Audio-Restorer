
from unittest.mock import MagicMock, patch
from pathlib import Path

# Use string based import/patching to handle previously imported modules
import restore_audio_hybrid


def test_ultra_profile_runs_parallel():
    """Test that ULTRA profile (>=22GB VRAM) uses ThreadPoolExecutor for parallel chains."""

    # Setup Mocks
    mock_executor = MagicMock()
    mock_future_vocals = MagicMock()
    mock_future_music = MagicMock()

    # Configure futures to return dummy paths
    mock_future_vocals.result.return_value = Path("enhanced_vocals.wav")
    mock_future_music.result.return_value = Path("denoised_music.wav")

    mock_executor.submit.side_effect = [mock_future_vocals, mock_future_music]
    # Context manager mock
    mock_executor.__enter__.return_value = mock_executor

    video_path = Path("test_video.mp4")
    target_dir = Path("output")

    # Patch dependencies
    with patch("restore_audio_hybrid.GPU_VRAM_GB", 24.0), \
         patch("concurrent.futures.ThreadPoolExecutor",
               return_value=mock_executor) as MockExecutor, \
         patch("restore_audio_hybrid._separate_stems_step",
               return_value=(Path("v.wav"), Path("b.wav"))), \
         patch("restore_audio_hybrid._enhance_vocals_step") as mock_vocal_step, \
         patch("restore_audio_hybrid._denoise_music_step") as mock_music_step, \
         patch("restore_audio_hybrid._extract_audio_step"), \
         patch("restore_audio_hybrid._align_stems"), \
         patch("restore_audio_hybrid._final_mix_step"), \
         patch("restore_audio_hybrid.is_valid_audio", return_value=False), \
         patch("restore_audio_hybrid.shutil.rmtree"):

        # Execute
        result = restore_audio_hybrid.process_hybrid_audio(video_path, "RTX 4090", target_dir)

        # Verify
        assert result is True

        # Check that ThreadPoolExecutor was initialized
        MockExecutor.assert_called_with(max_workers=2)

        # Check that jobs were submitted
        assert mock_executor.submit.call_count == 2

        # Verify specific calls
        submitted_funcs = [call_args[0][0] for call_args in mock_executor.submit.call_args_list]
        assert mock_vocal_step in submitted_funcs
        assert mock_music_step in submitted_funcs


def test_low_profile_runs_sequential():
    """Test that LOW profile (<22GB VRAM) runs sequentially."""

    video_path = Path("test_video.mp4")
    target_dir = Path("output")

    with patch("restore_audio_hybrid.GPU_VRAM_GB", 10.0), \
         patch("concurrent.futures.ThreadPoolExecutor") as MockExecutor, \
         patch("restore_audio_hybrid._extract_audio_step"), \
         patch("restore_audio_hybrid._separate_stems_step",
               return_value=(Path("v.wav"), Path("b.wav"))), \
         patch("restore_audio_hybrid._enhance_vocals_step") as mock_enh_voc, \
         patch("restore_audio_hybrid._denoise_music_step") as mock_den_mus, \
         patch("restore_audio_hybrid._align_stems"), \
         patch("restore_audio_hybrid._final_mix_step"), \
         patch("restore_audio_hybrid.is_valid_audio", return_value=False), \
         patch("restore_audio_hybrid.shutil.rmtree"):

        # Execute
        result = restore_audio_hybrid.process_hybrid_audio(video_path, "RTX 3080", target_dir)

        assert result is True

        # Ensure Executor was NOT used
        MockExecutor.assert_not_called()

        # Ensure individual steps were called
        mock_enh_voc.assert_called()
        mock_den_mus.assert_called()
