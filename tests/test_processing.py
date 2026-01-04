import sys
from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path
import modules.processing
import modules.utils
import shutil

# ---------------------------------------------------------
# Processing Logic
# ---------------------------------------------------------


@patch("modules.processing.is_valid_audio", return_value=True)
@patch("modules.processing.is_valid_video", return_value=True)
def test_process_skip(mock_valid_vid, mock_valid_aud, tmp_path):
    """Test process skips when valid audio exists."""
    vid = tmp_path / "vid.mp4"
    vid.write_text("x")
    assert modules.processing.process_hybrid_audio(vid, "GPU", target_output_dir=tmp_path) is True


@patch("modules.processing.attempt_cpu_run_with_retry")
@patch("modules.processing.is_valid_audio")
def test_extract_audio_step(mock_valid, mock_retry, tmp_path):
    mock_valid.side_effect = iter([False] + [True] * 20)
    """Test audio extraction step."""
    video = tmp_path / "video.mp4"
    video.write_text("video data")
    output = tmp_path / "audio.wav"

    # Fix: Create tmp file via side_effect so rename works
    def create_tmp(*args, **kwargs):
        tmp_wav = output.with_suffix(".tmp.wav")
        tmp_wav.write_text("fake audio")
        return True

    mock_retry.side_effect = create_tmp

    modules.processing._extract_audio_step(video, output)
    mock_retry.assert_called_once()


@patch("modules.processing.attempt_cpu_run_with_retry")
@patch("modules.processing.is_valid_audio", return_value=True)
def test_extract_audio_step_skip(mock_valid, mock_retry, tmp_path, capsys):
    """Test audio extraction step skips when valid audio exists."""
    video = tmp_path / "video.mp4"
    video.write_text("video data")
    output = tmp_path / "audio.wav"
    output.write_text("existing audio")

    modules.processing._extract_audio_step(video, output)
    mock_retry.assert_not_called()


@patch("modules.processing.attempt_run_with_retry")
@patch("modules.processing.is_valid_audio")
def test_separate_stems_step(mock_valid, mock_retry, tmp_path):
    mock_valid.side_effect = iter([False] + [True] * 20)
    """Test separate stems step."""
    audio = tmp_path / "audio.wav"
    audio.write_text("audio data")
    out_dir = tmp_path / "sep_out"
    out_dir.mkdir()

    # Create expected output files
    vocal_file = out_dir / "audio_(Vocals)_model_bs_roformer_test.wav"
    vocal_file.write_text("vocals")
    back_file = out_dir / "audio_(Instrumental)_model_bs_roformer_test.wav"
    back_file.write_text("background")

    result = modules.processing._separate_stems_step(audio, out_dir)
    mock_retry.assert_called_once()
    assert len(result) == 2
    assert result[0] == vocal_file
    # Output should have been renamed to Background
    assert "(Background)" in result[1].name
    assert result[1].exists()
    assert not back_file.exists()  # Old file should be gone


@patch("modules.processing.attempt_run_with_retry")
@patch("modules.processing.is_valid_audio", return_value=True)
def test_separate_stems_step_skip(mock_valid, mock_retry, tmp_path):
    """Test stem separation skips when valid stems exist."""
    audio = tmp_path / "audio.wav"
    audio.write_text("d")
    out_dir = tmp_path / "sep_out"
    out_dir.mkdir()

    # Create both expected output files
    vocal_file = out_dir / "audio_(Vocals)_model_bs_roformer_test.wav"
    vocal_file.write_text("v")
    back_file = out_dir / "audio_(Instrumental)_model_bs_roformer_test.wav"
    back_file.write_text("b")

    result = modules.processing._separate_stems_step(audio, out_dir)
    mock_retry.assert_not_called()

    # Should have been renamed
    renamed_back = out_dir / "audio_(Background)_model_bs_roformer_test.wav"
    assert result[0] == vocal_file
    assert result[1] == renamed_back
    assert renamed_back.exists()
    assert not back_file.exists()


@patch("modules.processing.run_command_with_progress")
@patch("modules.processing.shutil.copy")
@patch("modules.processing.shutil.rmtree")
@patch("modules.processing.is_valid_audio", return_value=False)
def test_enhance_vocals_step(mock_valid, mock_rm, mock_cp, mock_run, tmp_path):
    """Test vocal enhancement step."""
    vocals = tmp_path / "vocals.wav"
    vocals.write_text("vocals data")
    out_dir = tmp_path / "enhanced"
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    # Create expected output
    out_dir.mkdir(exist_ok=True)
    enhanced = out_dir / "enhanced_vocals.wav"
    enhanced.write_text("enhanced")

    modules.processing._enhance_vocals_step(vocals, out_dir, work_dir)
    mock_run.assert_called_once()


@patch("modules.processing.run_command_with_progress")
@patch("modules.processing.shutil.copy")
@patch("modules.processing.is_valid_audio", return_value=True)
def test_enhance_vocals_step_skip(mock_valid, mock_cp, mock_run, tmp_path):
    """Test enhance step skips when valid enhanced audio exists."""
    vocals = tmp_path / "vocals.wav"
    vocals.write_text("vocals")
    out_dir = tmp_path / "enhanced"
    out_dir.mkdir()
    existing = out_dir / "already_enhanced.wav"
    existing.write_text("already enhanced")
    work_dir = tmp_path / "work"

    result = modules.processing._enhance_vocals_step(vocals, out_dir, work_dir)
    mock_run.assert_not_called()
    assert result == existing


@patch("modules.processing.run_command_with_progress")
@patch("modules.processing.shutil.copy")
@patch("modules.processing.shutil.rmtree")
@patch("modules.processing.is_valid_audio", return_value=False)
def test_enhance_vocals_step_fallback(mock_valid, mock_rm, mock_cp, mock_run, tmp_path):
    """Test enhance step fallback when no output is produced."""
    vocals = tmp_path / "vocals.wav"
    vocals.write_text("vocals data")
    out_dir = tmp_path / "enhanced"
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    # Don't create output - should trigger fallback
    out_dir.mkdir(exist_ok=True)
    # No files in out_dir

    result = modules.processing._enhance_vocals_step(vocals, out_dir, work_dir)
    # Should have used the fallback copy
    assert mock_cp.called
    assert "fallback" in str(result)


@patch("modules.processing.attempt_run_with_retry")
@patch("modules.processing.is_valid_audio", return_value=False)
def test_denoise_background_step(mock_valid, mock_retry, tmp_path):
    """Test background denoising step."""
    background = tmp_path / "background.wav"
    background.write_text("background data")
    out_dir = tmp_path / "denoised"
    out_dir.mkdir()

    # Create expected output
    denoised = out_dir / "background_(No Noise).wav"
    denoised.write_text("denoised")

    # Fix: Mock side_effect: [False (initial), True (found denoise)]
    mock_valid.side_effect = iter([False] + [True] * 20)

    # Mock success and verify command builder
    def side_effect(*args, **kwargs):
        # Extract builder from args (pos 0)
        builder = args[0]
        # Verify builder
        cmd = builder(1)
        assert "--single_stem" in cmd
        assert "No Noise" in cmd
        return True

    mock_retry.side_effect = side_effect

    result = modules.processing._denoise_background_step(background, out_dir)
    assert "No Noise" in result.name


@patch("modules.processing.attempt_run_with_retry")
@patch("modules.processing.is_valid_audio", return_value=False)
def test_denoise_background_step_no_noise_selection(mock_valid, mock_retry, tmp_path):
    """Test denoise step selects (No Noise) variant."""
    background = tmp_path / "background.wav"
    background.write_text("background data")
    out_dir = tmp_path / "denoised"
    out_dir.mkdir()

    # Create both regular and "No Noise" outputs
    regular = out_dir / "background_(Background)_denoised.wav"
    regular.write_text("regular")
    no_noise = out_dir / "background_(Background)_(No Noise).wav"
    no_noise.write_text("no noise")

    result = modules.processing._denoise_background_step(background, out_dir)
    # Should prefer the (No Noise) version
    assert "(No Noise)" in str(result)


@patch("modules.processing.attempt_run_with_retry")
@patch("modules.processing.is_valid_audio", return_value=False)
def test_denoise_background_step_fallback_to_regular(mock_valid, mock_retry, tmp_path):
    """Test denoise step falls back to regular when no (No Noise) exists."""
    background = tmp_path / "background.wav"
    background.write_text("background data")
    out_dir = tmp_path / "denoised"
    out_dir.mkdir()

    # Create only regular output (no "No Noise" variant)
    regular = out_dir / "background_(Background)_denoised.wav"
    regular.write_text("regular")

    result = modules.processing._denoise_background_step(background, out_dir)
    assert result == regular


@patch("modules.processing.attempt_run_with_retry")
@patch("modules.processing.is_valid_audio", return_value=False)
@patch("modules.processing.log_msg")
def test_denoise_background_step_fallback_to_raw(mock_log, mock_valid, mock_retry, tmp_path):
    """Test denoise step falls back to raw background when denoising fails."""
    background = tmp_path / "background.wav"
    background.write_text("background data")
    out_dir = tmp_path / "denoised"
    out_dir.mkdir()

    # No output files - should fall back to raw
    result = modules.processing._denoise_background_step(background, out_dir)
    assert result == background
    mock_log.assert_called()


@patch("modules.processing.attempt_run_with_retry")
@patch("modules.processing.is_valid_audio", side_effect=[True])
def test_denoise_background_step_skip(mock_valid, mock_retry, tmp_path):
    """Test denoise step skips when valid denoised exists."""
    background = tmp_path / "background.wav"
    background.write_text("background")
    out_dir = tmp_path / "denoised"
    out_dir.mkdir()

    # Create existing valid denoised file
    existing = out_dir / "background_(Background)_denoised.wav"
    existing.write_text("denoised")

    result = modules.processing._denoise_background_step(background, out_dir)
    mock_retry.assert_not_called()
    assert result == existing


@patch("modules.processing.attempt_cpu_run_with_retry")
@patch("modules.processing.get_audio_duration_sec", return_value=60.0)
@patch("modules.processing.is_valid_video")
def test_final_mix_step(mock_is_valid_vid, mock_dur, mock_retry, tmp_path):
    mock_is_valid_vid.side_effect = iter([False] + [True] * 20)
    """Test final mix step."""
    video = tmp_path / "video.mp4"
    video.write_text("video")
    vocals = tmp_path / "vocals.wav"
    vocals.write_text("vocals")
    background = tmp_path / "background.wav"
    background.write_text("background")
    output = tmp_path / "final.mp4"

    def create_tmp(*args, **kwargs):
        tmp_mp4 = output.with_suffix(".tmp.mp4")
        tmp_mp4.write_text("fake video")
        return True

    mock_retry.side_effect = create_tmp

    modules.processing._final_mix_step(video, vocals, background, output)
    mock_retry.assert_called_once()


@patch("modules.processing._extract_audio_step")
@patch("modules.processing._separate_stems_step")
@patch("modules.processing._enhance_vocals_step")
@patch("modules.processing._denoise_background_step")
@patch("modules.sync._align_stems")
@patch("modules.processing._final_mix_step")
@patch("modules.processing.is_valid_audio", return_value=False)
@patch("modules.processing.shutil.rmtree")
@patch("modules.processing.shutil.copy")
def test_full_pipeline_coverage(mc, mr, mv, s6, s5, s4, s3, s2, s1, tmp_path):
    """Test full pipeline coverage."""
    v = tmp_path / "v.mp4"
    v.write_text("d")
    out = tmp_path / "out"
    out.mkdir()

    # Mock _separate_stems_step return
    s2.return_value = (Path("vocals.wav"), Path("back.wav"))

    assert modules.processing.process_hybrid_audio(v, "GPU", out) is True

    # Re-run (resume) handled by test_process_skip but here for branch coverage in main func?
    # No, test_process_skip mocks is_valid_video for OUTPUT.
    # Here default is_valid mock might need to toggle to cover resume.
    # But assertions are simple.
    assert modules.processing.process_hybrid_audio(v, "GPU", out) is True


# Removed test_full_pipeline_denoise_only_mode as logic is pending reimplementation.


@patch("soundfile.SoundFile")
def test_get_audio_duration_sec(mock_sf):
    """Test audio duration calculation."""
    mock_ctx = MagicMock()
    mock_ctx.frames = 44100 * 10  # 10 seconds
    mock_ctx.samplerate = 44100
    mock_sf.return_value.__enter__.return_value = mock_ctx

    duration = modules.processing.get_audio_duration_sec(Path("test.wav"))
    assert duration == 10.0

    # Test exception handling
    mock_sf.side_effect = Exception("File not found")
    duration = modules.processing.get_audio_duration_sec(Path("nonexistent.wav"))
    # Note: exception in constructor means it raises before enter.
    assert duration is None


@patch("modules.utils.is_valid_video", return_value=False)
@patch("modules.processing.get_video_duration_sec", return_value=10.0)
@patch("modules.processing._extract_audio_step")
@patch("modules.processing._separate_stems_step")
@patch("modules.processing._enhance_vocals_step")
@patch("modules.processing._denoise_background_step")
@patch("modules.sync._align_stems")
@patch("modules.processing._final_mix_step")
def test_process_hybrid_audio_preservation(
    mock_mix, mock_align, mock_denoise, mock_enhance, mock_sep, mock_ext, mock_dur, mock_valid, tmp_path, capsys
):
    """Test preservation of work dir on verification failure."""
    video = tmp_path / "v.mp4"
    video.write_text("v")

    # Validation Sequence:
    # 1. Initial check (False -> proceed)
    # 2. Final cleanup check (False -> preserve)
    mock_valid.side_effect = [False, False]

    # Mock steps to return dummy paths
    mock_sep.return_value = (Path("v.wav"), Path("b.wav"))
    mock_enhance.return_value = Path("ev.wav")
    mock_denoise.return_value = Path("db.wav")

    # Enable Debug Logging for this test to capture the preservation message
    # Mocking modules.utils.DEBUG_LOGGING because log_msg is in utils and imports it.
    with patch("modules.utils.DEBUG_LOGGING", True):
        modules.processing.process_hybrid_audio(video, "GPU")

    captured = capsys.readouterr()
    assert "Preservation: Keeping" in captured.out


@patch("modules.processing._extract_audio_step")
@patch("modules.processing._separate_stems_step", return_value=(Path("v.wav"), Path("b.wav")))
@patch("modules.processing._enhance_vocals_step", return_value=Path("ev.wav"))
@patch("modules.processing._denoise_background_step", return_value=Path("db.wav"))
@patch("modules.sync._align_stems")
@patch("modules.processing._final_mix_step")
@patch("modules.processing.shutil.rmtree", side_effect=OSError("Access Denied"))
@patch("modules.processing.is_valid_video")
def test_process_rmtree_errors(mock_valid, mock_rmtree, mock_mix, mock_align, mock_den, mock_enh, mock_sep, mock_ext, tmp_path):
    """Test rmtree failure handling."""
    video = tmp_path / "v.mp4"
    video.write_text("v")
    # Mock video duration
    with patch("modules.processing.get_video_duration_sec", return_value=10.0):
        work_dir = video.parent / f".temp_work_{video.stem}"
        work_dir.mkdir(parents=True)

        # Validation: Start->False, Final->True (to trigger cleanup)
        mock_valid.side_effect = [False, True]

        res = modules.processing.process_hybrid_audio(video, "GPU")
        assert res is True
        # If rmtree raised OSError, it should be caught in finally.
        # So no exception raised here.


def test_final_mix_missing_vocals(tmp_path):
    """Test final mix raises FileNotFoundError if vocals missing."""
    video = tmp_path / "v.mp4"
    voc = tmp_path / "v.wav"
    bg = tmp_path / "b.wav"
    out = tmp_path / "out.mp4"
    # Don't create files
    
    with pytest.raises(FileNotFoundError):
        modules.processing._final_mix_step(video, voc, bg, out)


def test_process_hybrid_audio_file_not_found(tmp_path, capsys):
    """Test process returns False if video file not found."""
    video = tmp_path / "nonexistent.mp4"
    res = modules.processing.process_hybrid_audio(video, "GPU")
    assert res is False
    captured = capsys.readouterr()
    assert "[Error] File not found" in captured.out


@patch("modules.processing._extract_audio_step", side_effect=Exception("Step Failed"))
@patch("modules.processing.get_video_duration_sec", return_value=10.0)
def test_process_hybrid_audio_error_handling(mock_dur, mock_ext, tmp_path, capsys):
    """Test process catches exception and returns False."""
    video = tmp_path / "v.mp4"
    video.write_text("v")
    res = modules.processing.process_hybrid_audio(video, "GPU")
    assert res is False
    captured = capsys.readouterr()
    assert "[Error] Processing failed: Step Failed" in captured.out


@patch("modules.processing.attempt_cpu_run_with_retry")
@patch("modules.processing.is_valid_video", return_value=False) # Output invalid
@patch("modules.processing.get_audio_duration_sec", return_value=123)
def test_final_mix_step_failure(mock_dur, mock_valid, mock_retry, tmp_path):
    """Test final mix raises exception if output is invalid."""
    video = tmp_path / "v.mp4"
    voc = tmp_path / "v.wav"
    voc.touch()
    bg = tmp_path / "b.wav"
    bg.touch()
    out = tmp_path / "out.mp4"
    
    with pytest.raises(Exception, match="Final Mix Failed"):
        modules.processing._final_mix_step(video, voc, bg, out)
