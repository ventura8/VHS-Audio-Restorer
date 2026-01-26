from unittest.mock import patch
from pathlib import Path
import restore_audio_hybrid


def test_main_success():
    """Test main execution path with valid inputs."""
    mock_input_files = ([Path("test.mp4")], False)

    with patch("restore_audio_hybrid.run_init_sequence", return_value=("CPU", "GPU")), \
         patch("restore_audio_hybrid.check_dependencies", return_value=True), \
         patch("restore_audio_hybrid._show_banner"), \
         patch("restore_audio_hybrid.OUTPUT_DIR") as mock_output_dir, \
         patch("restore_audio_hybrid._get_input_files", return_value=mock_input_files), \
         patch("restore_audio_hybrid.process_hybrid_audio") as mock_process, \
         patch("builtins.input"):

        restore_audio_hybrid.main()

        mock_output_dir.mkdir.assert_called_once()
        mock_process.assert_called_once()
        args, kwargs = mock_process.call_args
        assert args[0] == Path("test.mp4")
        assert args[1] == "GPU"


def test_main_no_dependencies():
    """Test main exits if dependencies missing."""
    with patch("restore_audio_hybrid.run_init_sequence", return_value=("CPU", "GPU")), \
         patch("restore_audio_hybrid.check_dependencies", return_value=False), \
         patch("builtins.print") as mock_print:

        restore_audio_hybrid.main()

        # Should print error and return
        assert any("Critical Error" in str(c) for c in mock_print.call_args_list)


def test_main_no_files():
    """Test main exits if no files selected."""
    with patch("restore_audio_hybrid.run_init_sequence", return_value=("CPU", "GPU")), \
         patch("restore_audio_hybrid.check_dependencies", return_value=True), \
         patch("restore_audio_hybrid._show_banner"), \
         patch("restore_audio_hybrid._show_banner"), \
         patch("restore_audio_hybrid.OUTPUT_DIR"), \
         patch("restore_audio_hybrid._get_input_files", return_value=([], False)), \
         patch("builtins.print") as mock_print:

        restore_audio_hybrid.main()
        assert any("No valid video files found" in str(c) for c in mock_print.call_args_list)


def test_main_keyboard_interrupt():
    """Test main handles interruptions graciously."""
    mock_input_files = ([Path("test.mp4")], False)

    with patch("restore_audio_hybrid.run_init_sequence", return_value=("CPU", "GPU")), \
         patch("restore_audio_hybrid.check_dependencies", return_value=True), \
         patch("restore_audio_hybrid._show_banner"), \
         patch("restore_audio_hybrid.OUTPUT_DIR"), \
         patch("restore_audio_hybrid._get_input_files", return_value=mock_input_files), \
         patch("restore_audio_hybrid.process_hybrid_audio"), \
         patch("builtins.input", side_effect=KeyboardInterrupt):
        restore_audio_hybrid.main()
