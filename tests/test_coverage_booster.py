import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import modules.processing
import modules.hardware
import os
import shutil

def test_extract_audio_step_cleanup_on_fail(tmp_path):
    """Cover lines 79-80 and 81 in processing.py"""
    video = tmp_path / "v.mp4"
    video.touch()
    out = tmp_path / "out.wav"
    
    # Mock is_valid_audio to return False for the tmp file
    with patch("modules.processing.is_valid_audio", return_value=False):
        with patch("modules.processing.attempt_cpu_run_with_retry") as mock_retry:
            def create_bad_tmp(*args, **kwargs):
                tmp = out.with_suffix(".tmp.wav")
                tmp.write_text("corrupt")
                return True
            mock_retry.side_effect = create_bad_tmp
            
            with pytest.raises(Exception, match="Extraction failed"):
                modules.processing._extract_audio_step(video, out)
            
            assert not out.with_suffix(".tmp.wav").exists()

def test_extract_audio_step_existing_unlink(tmp_path):
    """Cover line 54 in processing.py"""
    video = tmp_path / "v.mp4"
    video.touch()
    out = tmp_path / "out.wav"
    out.touch()
    
    # is_valid_audio must return False for "out" to trigger unlink, but we need to mock it carefully
    # 1. Initial check (is_valid_audio(original_wav)) -> returns False
    # 2. Final check (is_valid_audio(tmp_wav)) -> returns True
    with patch("modules.processing.is_valid_audio", side_effect=[False, True]):
        with patch("modules.processing.attempt_cpu_run_with_retry") as mock_retry:
            def create_tmp(*args, **kwargs):
                tmp = out.with_suffix(".tmp.wav")
                tmp.write_text("data")
                return True
            mock_retry.side_effect = create_tmp
            
            modules.processing._extract_audio_step(video, out)
            assert out.exists()

def test_verify_separation_output_fallback(tmp_path):
    """Cover lines 95-97 in processing.py"""
    sep_dir = tmp_path / "sep"
    sep_dir.mkdir()
    v = sep_dir / "vocals.wav"
    v.write_text("v")
    b = sep_dir / "other.wav" # No (Instrumental) or (Background) tag
    b.write_text("b")
    
    with patch("modules.processing.is_valid_audio", return_value=True):
        # We need to ensure glob finds them.
        # list(separation_out_dir.glob("*(Vocals)*.wav")) -> should find v if we name it right
        v.rename(sep_dir / "test_(Vocals).wav")
        v = sep_dir / "test_(Vocals).wav"
        
        vocals, background = modules.processing._verify_separation_output(sep_dir, Path("orig.wav"))
        assert vocals == v
        assert background.name == "test_(Background).wav"

def test_separate_stems_step_debug_logging(tmp_path):
    """Cover lines 181-183 in processing.py"""
    audio = tmp_path / "audio.wav"
    audio.touch()
    sep_dir = tmp_path / "sep"
    sep_dir.mkdir()
    
    with patch("modules.processing._verify_separation_output", return_value=(None, None)):
        mock_sep_module = MagicMock()
        with patch.dict("sys.modules", {"audio_separator": MagicMock(), "audio_separator.separator": mock_sep_module}):
            mock_sep = mock_sep_module.Separator.return_value
            # Return 2 files but _verify still returns None (simulated mismatch)
            mock_sep.separate.return_value = ["v.wav", "b.wav"]
            
            with patch("modules.processing.log_msg") as mock_log:
                with pytest.raises(Exception, match="output stems were not identified"):
                    modules.processing._separate_stems_step(audio, sep_dir)
                # Check if debug log was called for line 182
                mock_log.assert_any_call("    [Debug] Separator returned: ['v.wav', 'b.wav']", level="DEBUG")

def test_run_enhance_retry_coverage():
    """Cover lines 202-212 in processing.py"""
    import subprocess
    cmd = ["false"]
    
    with patch("modules.processing.run_command_with_progress", side_effect=subprocess.CalledProcessError(1, cmd)):
        with patch("modules.processing.time.sleep"):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.empty_cache") as mock_empty:
                    with pytest.raises(subprocess.CalledProcessError):
                        modules.processing._run_enhance_retry(cmd, 10)
                    assert mock_empty.called

def test_enhance_vocals_cleanup_errors(tmp_path):
    """Cover lines 249, 257, 288-289 in processing.py"""
    voc = tmp_path / "v.wav"
    voc.touch()
    enh_dir = tmp_path / "enh"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    
    # Mock rmtree to fail for line 275 coverage (though it has pragma: no cover, let's hit it)
    with patch("modules.processing.shutil.rmtree", side_effect=Exception("rm error")):
        with patch("modules.processing.shutil.copy"):
            with patch("modules.processing._run_enhance_retry"):
                with patch("modules.processing._handle_enhance_output", return_value=Path("dummy.wav")):
                    # This should not raise because of try-except
                    modules.processing._enhance_vocals_step(voc, enh_dir, work_dir)

def test_denoise_background_warning(tmp_path):
    """Cover lines 340-341 in processing.py"""
    bg = tmp_path / "bg.wav"
    bg.touch()
    den_dir = tmp_path / "den"
    den_dir.mkdir()
    
    mock_sep_module = MagicMock()
    with patch.dict("sys.modules", {"audio_separator": MagicMock(), "audio_separator.separator": mock_sep_module}):
        mock_sep = mock_sep_module.Separator.return_value
        mock_sep.separate.return_value = []
        
        # Mock glob to return empty lists for both checks
        with patch("pathlib.Path.glob", return_value=[]):
            with patch("modules.processing.log_msg") as mock_log:
                res = modules.processing._denoise_background_step(bg, den_dir)
                assert res == bg
                mock_log.assert_any_call("    [Warning] UVR-DeNoise failed. Using raw background.", is_error=True)

def test_final_mix_step_missing_background(tmp_path):
    """Cover line 369 in processing.py"""
    v = tmp_path / "v.mp4"
    voc = tmp_path / "voc.wav"
    voc.touch()
    bg = tmp_path / "bg.wav"
    # bg doesn't exist
    out = tmp_path / "out.mp4"
    
    with pytest.raises(FileNotFoundError, match="Missing Background"):
        modules.processing._final_mix_step(v, voc, bg, out)

def test_final_mix_step_existing_unlink(tmp_path):
    """Cover line 402 in processing.py"""
    v = tmp_path / "v.mp4"
    voc = tmp_path / "voc.wav"
    voc.touch()
    bg = tmp_path / "bg.wav"
    bg.touch()
    out = tmp_path / "out.mp4"
    out.touch() # Existing output
    
    # Mock to pass verification
    with patch("modules.processing.is_valid_video", side_effect=[False, True]):
        with patch("modules.processing.attempt_cpu_run_with_retry") as mock_retry:
            def create_tmp(*args, **kwargs):
                tmp = out.with_suffix(".tmp.mp4")
                tmp.touch()
                return True
            mock_retry.side_effect = create_tmp
            
            modules.processing._final_mix_step(v, voc, bg, out)
            assert out.exists()

def test_process_hybrid_audio_denoise_only(tmp_path):
    """Cover line 455 in processing.py"""
    v = tmp_path / "v.mp4"
    v.touch()
    
    with patch("modules.processing.PROCESS_MODE", "denoise_only"):
        with patch("modules.processing.get_video_duration_sec", return_value=10):
            with patch("modules.processing._extract_audio_step"):
                with patch("modules.processing._separate_stems_step", return_value=(Path("v.wav"), Path("b.wav"))):
                    with patch("modules.processing._enhance_vocals_step", return_value=Path("ev.wav")):
                        with patch("modules.processing._denoise_background_step", return_value=Path("db.wav")):
                            with patch("modules.sync._align_stems"):
                                with patch("modules.processing._final_mix_step"):
                                    with patch("modules.processing.is_valid_video", side_effect=[False, True]):
                                        res = modules.processing.process_hybrid_audio(v, "GPU")
                                        assert res is True

# Hardware Coverage Booster
def test_get_gpu_name_pytorch_with_index():
    """Cover lines 114-119 in hardware.py"""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("modules.hardware.CUDA_DEVICE", "cuda:1"):
            with patch("torch.cuda.get_device_name") as mock_name:
                mock_name.return_value = "Test GPU 1"
                assert modules.hardware.get_gpu_name() == "Test GPU 1"
                mock_name.assert_called_with(1)

def test_get_nvidia_paths_comprehensive_booster():
    """Cover lines 145-162 in hardware.py."""
    m_lib = MagicMock()
    m_lib.__path__ = ["/fake/p"]
    
    # We mock 'nvidia' module and its submodules in sys.modules
    m_nvidia = MagicMock()
    m_nvidia.cudnn = m_lib
    m_nvidia.cublas = m_lib
    
    with patch.dict("sys.modules", {
        "nvidia": m_nvidia, 
        "nvidia.cudnn": m_lib, 
        "nvidia.cublas": m_lib
    }):
        with patch("os.path.exists", return_value=True):
            # We also need to ensure the local scope of get_nvidia_paths sees 'nvidia'
            # when it does 'for lib in [nvidia.cudnn, nvidia.cublas]'
            with patch("modules.hardware.nvidia", m_nvidia, create=True):
                res = modules.hardware.get_nvidia_paths()
                assert any("/fake/p" in p.replace("\\", "/") for p in res)

def test_get_gpu_name_nvidia_smi_success():
    """Cover lines 124-125 in hardware.py."""
    with patch("torch.cuda.is_available", return_value=False):
        with patch("subprocess.check_output") as mock_run:
            mock_run.return_value = b"GPU 0: NVIDIA Test GPU (UUID: 123)"
            name = modules.hardware.get_gpu_name()
            assert "NVIDIA Test GPU" in name

def test_get_gpu_name_pytorch_exception():
    """Cover lines 118-119 in hardware.py."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("modules.hardware.CUDA_DEVICE", "invalid:device"):
            # This should cause an exception in index parsing or similar
            name = modules.hardware.get_gpu_name()
            # It should fall back to nvidia-smi or generic
            assert name is not None
