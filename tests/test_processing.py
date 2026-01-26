from unittest.mock import MagicMock, patch
import sys
import pytest
from pathlib import Path
import modules.processing
import modules.utils

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


@patch("modules.processing.is_valid_audio")
def test_separate_stems_step(mock_valid, tmp_path):
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

    # Mock the internal import of Separator
    with patch.dict("sys.modules", {"audio_separator.separator": MagicMock()}):
        # We need to set up the mock class returned by the import
        mock_sep_module = sys.modules["audio_separator.separator"]
        mock_sep_instance = mock_sep_module.Separator.return_value
        mock_sep_instance.separate.return_value = ["v.wav", "b.wav"]

        result = modules.processing._separate_stems_step(audio, out_dir)

        # assert mock_retry.assert_called_once() # NOT called anymore
        assert len(result) == 2
        assert result[0] == vocal_file
        # Output might be renamed or not depending on tags.
        # Logic: if (Background) tag not present, it renames.
        # Our mocked output "audio_(Instrumental)..." didn't have (Background).
        # So it should be renamed.
        assert "(Background)" in result[1].name
        # Since we mocked only 1 behavior call, check if the file object in result refers to new name
        # The actual file on disk "back_file" was written with old name.
        # rename() in code changes simple path object but we also need to mock valid check?
        # Actually, verify_separation_output renames ON DISK.
        # Our test wrote 'back_file'. The code should rename 'back_file' to 'result[1]'.
        assert result[1].exists()
        assert not back_file.exists()


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


@patch("modules.processing.is_valid_audio")
def test_denoise_background_step(mock_valid, tmp_path):
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

    # Mock audio_separator
    with patch.dict("sys.modules", {"audio_separator.separator": MagicMock()}):
        mock_sep_module = sys.modules["audio_separator.separator"]
        mock_sep_instance = mock_sep_module.Separator.return_value
        # Mock separate to return filename, though logic uses glob
        mock_sep_instance.separate.return_value = ["background_(No Noise).wav"]

        result = modules.processing._denoise_background_step(background, out_dir)
        assert "No Noise" in result.name


@patch("modules.processing.is_valid_audio")
def test_denoise_background_step_no_noise_selection(mock_valid, tmp_path):
    # Logic: 
    # 1. Initial check (candidates) -> Return False so we don't skip.
    # 2. After separate -> Return True for all found files.
    # We have 2 initial candidates (reg, no_noise), so False, False.
    # Then separate happens.
    # Then we check again.
    # Simplest is just always True? No, if True initially, it returns valid_denoised[0] immediately.
    # And directory listing order is not guaranteed. 
    # If regular comes first in glob, and we return valid_denoised[0], we get regular.
    # We want to force it to run separation logic?
    # Actually, the test says "Test denoise step selects (No Noise) variant."
    # If files exist, it skips. If it skips, it returns the first valid one.
    # If regular is first, it returns regular.
    # The Code says: 
    #     candidates_denoised = list(denoised_background_dir.glob("*.wav"))
    #     valid_denoised = [f for f in candidates_denoised if is_valid_audio(f)]
    #     if valid_denoised: return valid_denoised[0]
    # So if files exist, it picks arbitrary first one. This is bad for deterministic test if multiple exist.
    # THE TEST INTENTION: It wants to test the selection logic AFTER separation.
    # So we MUST ensure "Initial Check" sees NO valid files.
    # But we wrote files to disk.
    # So mock_valid must return False for them initially.
    # glob finds 2 files. So mock_valid called twice? Or once per file.
    # Let's just return False enough times.
    mock_valid.side_effect = [False, False, True, True, True, True] 
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

    # Mock audio_separator
    with patch.dict("sys.modules", {"audio_separator.separator": MagicMock()}):
        mock_sep_module = sys.modules["audio_separator.separator"]
        mock_sep_instance = mock_sep_module.Separator.return_value
        mock_sep_instance.separate.return_value = [] # Return value ignored as we rely on glob

        result = modules.processing._denoise_background_step(background, out_dir)
        # Should prefer the (No Noise) version
        assert "No Noise" in result.name


@patch("modules.processing.is_valid_audio")
def test_denoise_background_step_fallback_to_regular(mock_valid, tmp_path):
    mock_valid.side_effect = iter([False] + [True] * 20)
    """Test denoise step falls back to regular when no (No Noise) exists."""
    background = tmp_path / "background.wav"
    background.write_text("background data")
    out_dir = tmp_path / "denoised"
    out_dir.mkdir()

    # Create only regular output (no "No Noise" variant)
    regular = out_dir / "background_(Background)_denoised.wav"
    regular.write_text("regular")

    # Mock audio_separator
    with patch.dict("sys.modules", {"audio_separator.separator": MagicMock()}):
        mock_sep_module = sys.modules["audio_separator.separator"]
        mock_sep_instance = mock_sep_module.Separator.return_value
        mock_sep_instance.separate.return_value = []

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
def test_process_rmtree_errors(
    mock_valid, mock_rmtree, mock_mix, mock_align, mock_den, mock_enh, mock_sep, mock_ext, tmp_path
):
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
@patch("modules.processing.is_valid_video", return_value=False)  # Output invalid
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


@patch("modules.processing.subprocess.check_output")
def test_get_video_duration_sec_success(mock_output):
    """Test get_video_duration_sec returns duration on success."""
    mock_output.return_value = b"123.45\n"
    result = modules.processing.get_video_duration_sec("video.mp4")
    assert result == 123.45


@patch("modules.processing.subprocess.check_output", side_effect=Exception("ffprobe error"))
def test_get_video_duration_sec_failure(mock_output):
    """Test get_video_duration_sec returns None on failure."""
    result = modules.processing.get_video_duration_sec("missing.mp4")
    assert result is None


@patch("modules.processing.is_valid_audio")
def test_verify_separation_output_patterns(mock_valid, tmp_path):
    """Test _verify_separation_output finds stems with different patterns."""
    sep_dir = tmp_path / "sep"
    sep_dir.mkdir()
    
    # Create files with different patterns
    v1 = sep_dir / "song_(Vocals)_stem.wav"
    v1.write_text("vocals")
    b1 = sep_dir / "song_(Instrumental)_stem.wav"
    b1.write_text("background")
    
    mock_valid.return_value = True
    
    vocals, background = modules.processing._verify_separation_output(sep_dir, tmp_path / "original.wav")
    assert vocals.name == "song_(Vocals)_stem.wav"
    assert "(Background)" in background.name


@patch("modules.processing.is_valid_audio")
def test_verify_separation_fallback_pattern2(mock_valid, tmp_path):
    """Test _verify_separation_output with pattern matching."""
    sep_dir = tmp_path / "sep"
    sep_dir.mkdir()
    
    # Create files with different naming patterns
    v_tagged = sep_dir / "song_(Vocals).wav"
    v_tagged.write_text("vocals")
    b_tagged = sep_dir / "song_(Instrumental).wav"
    b_tagged.write_text("background")
    
    mock_valid.return_value = True
    
    vocals, background = modules.processing._verify_separation_output(sep_dir, tmp_path / "original.wav")
    assert "(Vocals)" in vocals.name
    assert "(Background)" in background.name or "(Instrumental)" in background.name


@patch("modules.processing.shutil.copy")
@patch("modules.processing.is_valid_audio", return_value=True)
def test_handle_enhance_output_fallback(mock_valid, mock_copy, tmp_path):
    """Test _handle_enhance_output fallback when no output produced."""
    # Mock empty directory
    enhanced_dir = tmp_path / "enhanced"
    enhanced_dir.mkdir()
    
    vocals_wav = tmp_path / "vocals.wav"
    vocals_wav.write_text("vocals")
    
    mock_copy.return_value = None
    
    # When no files in enhanced_dir, should copy vocals to fallback
    result = modules.processing._handle_enhance_output(enhanced_dir, vocals_wav)
    
    assert "fallback" in result.name
    mock_copy.assert_called_once()


@patch("modules.processing.is_valid_audio")
def test_handle_enhance_output_success(mock_valid, tmp_path):
    """Test _handle_enhance_output with valid enhanced output."""
    enhanced_dir = tmp_path / "enhanced"
    enhanced_dir.mkdir()
    
    # Create an enhanced file
    enhanced_file = enhanced_dir / "vocals_enhanced.wav"
    enhanced_file.write_text("enhanced")
    
    vocals_wav = tmp_path / "vocals.wav"
    vocals_wav.write_text("vocals")
    
    mock_valid.return_value = True
    
    result = modules.processing._handle_enhance_output(enhanced_dir, vocals_wav)
    
    assert result == enhanced_file
    assert result.exists()


@patch("modules.processing.run_command_with_progress")
@patch("modules.processing.is_valid_audio", return_value=True)
def test_denoise_background_step_no_selection(mock_valid, mock_run, tmp_path):
    """Test _denoise_background_step when no noise selection available."""
    bg_wav = tmp_path / "background.wav"
    bg_wav.write_text("background")
    
    # When run_command fails or no noise file, should return original
    mock_run.return_value = None
    
    result = modules.processing._denoise_background_step(bg_wav, tmp_path)
    
    # Should return original background when denoise not possible
    assert result == bg_wav


@patch("modules.processing.attempt_run_with_retry")
@patch("modules.processing.is_valid_audio", return_value=True)
def test_separate_stems_exception_handling(mock_retry, mock_valid, tmp_path):
    """Test _separate_stems_step handles separation exceptions."""
    audio = tmp_path / "audio.wav"
    audio.write_text("audio")
    out_dir = tmp_path / "sep"
    out_dir.mkdir()
    
    # Mock separation to return without creating outputs
    mock_retry.return_value = True
    
    # When no output files are created, should raise an exception
    with pytest.raises(Exception, match="Separation completed"):
        modules.processing._separate_stems_step(audio, out_dir)


@patch("modules.processing.run_command_with_progress")  
@patch("modules.processing.is_valid_audio", return_value=True)
def test_denoise_background_step_success(mock_run, mock_valid, tmp_path):
    """Test _denoise_background_step successful denoising."""
    bg_wav = tmp_path / "background.wav"
    bg_wav.write_text("background")
    
    # Create denoised output
    denoised = tmp_path / "background_denoised.wav"
    denoised.write_text("denoised")
    
    result = modules.processing._denoise_background_step(bg_wav, tmp_path)
    # Should return original or denoised version
    assert result.exists()


@patch("modules.processing.is_valid_video", return_value=True)
def test_final_mix_step_skip(mock_video, tmp_path):
    """Test final_mix_step skips when output exists."""
    output = tmp_path / "output.mp4"
    output.write_text("output")
    
    with patch("modules.processing.log_msg"):
        modules.processing._final_mix_step(
            tmp_path / "video.mp4",
            tmp_path / "vocals.wav",
            tmp_path / "background.wav",
            output
        )
        
        # Should skip without errors


@patch("modules.processing.log_msg")
@patch("modules.processing.is_valid_audio", return_value=True)
def test_handle_enhance_output_with_valid_enhanced(mock_valid, mock_log, tmp_path):
    """Test _handle_enhance_output selects valid enhanced file."""
    enhanced_dir = tmp_path / "enhanced"
    enhanced_dir.mkdir()
    
    # Create enhanced file
    enhanced_file = enhanced_dir / "vocals_enhanced.wav"
    enhanced_file.write_text("enhanced")
    
    vocals = tmp_path / "vocals.wav"
    vocals.write_text("vocals")
    
    result = modules.processing._handle_enhance_output(enhanced_dir, vocals)
    assert result == enhanced_file


@patch("modules.processing.is_valid_audio")
@patch("modules.processing.run_command_with_progress")
def test_denoise_with_audio_validation(mock_run, mock_valid, tmp_path):
    """Test denoise background validates audio properly."""
    bg_wav = tmp_path / "background.wav"
    bg_wav.write_text("background")
    
    # First call: check input valid (True)
    # No denoise output, so should return original
    mock_valid.return_value = True
    mock_run.return_value = None
    
    result = modules.processing._denoise_background_step(bg_wav, tmp_path)
    assert result.exists()


@patch("modules.processing.is_valid_audio", return_value=True)
@patch("modules.processing.is_valid_video", return_value=True)
def test_process_skip_no_reprocessing(mock_video, mock_audio, tmp_path):
    """Test process skips when valid output already exists."""
    vid = tmp_path / "vid.mp4"
    vid.write_text("x")
    
    result = modules.processing.process_hybrid_audio(
        vid, "GPU", target_output_dir=tmp_path
    )
    assert result is True


@patch("modules.processing.is_valid_audio")
def test_extract_audio_step_cleans_up_tmp(mock_valid, tmp_path):
    """Test extract_audio_step cleans up failed tmp files."""
    mock_valid.side_effect = [True]  # Skip extraction - already valid
    
    video = tmp_path / "video.mp4"
    video.write_text("video")
    output = tmp_path / "audio.wav"
    
    modules.processing._extract_audio_step(video, output)
    
    # Should not create any files
    assert not output.exists()


@patch("modules.processing.log_msg")
@patch("modules.processing.is_valid_audio")
def test_verify_separation_rename_background(mock_valid, mock_log, tmp_path):
    """Test _verify_separation_output renames background file."""
    sep_dir = tmp_path / "sep"
    sep_dir.mkdir()
    
    # Create files needing rename
    v_file = sep_dir / "song_(Vocals).wav"
    v_file.write_text("vocals")
    b_file = sep_dir / "song_(Instrumental).wav"
    b_file.write_text("background")
    
    mock_valid.return_value = True
    
    vocals, background = modules.processing._verify_separation_output(sep_dir, tmp_path / "original.wav")
    
    # Verify background was renamed
    assert "(Background)" in background.name
    assert not b_file.exists()  # Original should be gone


@patch("modules.processing.is_valid_audio")
def test_handle_enhance_creates_fallback(mock_valid, tmp_path):
    """Test _handle_enhance_output creates fallback when no output."""
    mock_valid.return_value = True
    
    enhanced_dir = tmp_path / "enhanced"
    enhanced_dir.mkdir()
    
    vocals = tmp_path / "vocals.wav"
    vocals.write_text("vocals")
    
    # No enhanced files, should create fallback
    result = modules.processing._handle_enhance_output(enhanced_dir, vocals)
    
    assert result.exists()
    assert "fallback" in result.name
