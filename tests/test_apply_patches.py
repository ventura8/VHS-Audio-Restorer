"""
Tests for apply_patches.py
Uses real file fixtures to achieve actual code coverage.
"""
import os
import pytest

import apply_patches


@pytest.fixture
def mock_venv(tmp_path, monkeypatch):
    """Create a mock venv structure and chdir to tmp_path."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)

    # Create base venv structure
    venv_site = tmp_path / "venv" / "Lib" / "site-packages"
    venv_site.mkdir(parents=True)

    yield tmp_path, venv_site

    os.chdir(old_cwd)


# ---------------------------------------------------------
# patch_resemble_enhance Tests
# ---------------------------------------------------------

def test_patch_resemble_enhance_no_venv(tmp_path, capsys):
    """Test when venv doesn't exist."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        apply_patches.patch_resemble_enhance()
        captured = capsys.readouterr()
        assert "venv site-packages not found" in captured.out
    finally:
        os.chdir(old_cwd)


def test_patch_resemble_enhance_no_package(mock_venv, capsys):
    """Test when resemble_enhance package doesn't exist."""
    tmp_path, venv_site = mock_venv
    apply_patches.patch_resemble_enhance()
    captured = capsys.readouterr()
    assert "resemble_enhance package not found" in captured.out


def test_patch_resemble_enhance_already_patched_v2(mock_venv, capsys):
    """Test when files are already patched with V2."""
    tmp_path, venv_site = mock_venv

    resemble_dir = venv_site / "resemble_enhance"
    resemble_dir.mkdir()
    enhancer_dir = resemble_dir / "enhancer"
    enhancer_dir.mkdir()

    # File already patched with V2
    test_file = resemble_dir / "module.py"
    test_file.write_text("# PATCHED V2\nimport deepspeed\n", encoding="utf-8")

    # __main__.py with custom_load already present
    main_py = enhancer_dir / "__main__.py"
    main_py.write_text("import torchaudio\ncustom_load = lambda: None\n", encoding="utf-8")

    apply_patches.patch_resemble_enhance()
    captured = capsys.readouterr()
    # Should silently skip already patched files
    assert "already monkeypatched" in captured.out


def test_patch_resemble_enhance_deepspeed_patch(mock_venv, capsys):
    """Test actual DeepSpeed patching."""
    tmp_path, venv_site = mock_venv

    resemble_dir = venv_site / "resemble_enhance"
    resemble_dir.mkdir()
    enhancer_dir = resemble_dir / "enhancer"
    enhancer_dir.mkdir()

    # Create file needing patching
    test_file = resemble_dir / "model.py"
    test_file.write_text(
        "import deepspeed\nfrom deepspeed import config\nclass Model: pass\n",
        encoding="utf-8"
    )

    # Create __main__.py needing torchaudio patch
    main_py = enhancer_dir / "__main__.py"
    main_py.write_text("import torchaudio\ndef main(): pass\n", encoding="utf-8")

    apply_patches.patch_resemble_enhance()

    # Verify patching occurred
    patched = test_file.read_text(encoding="utf-8")
    assert "MockDeepSpeed" in patched
    assert "# PATCHED" in patched

    main_patched = main_py.read_text(encoding="utf-8")
    assert "custom_load" in main_patched

    captured = capsys.readouterr()
    assert "Patching" in captured.out


def test_patch_resemble_enhance_no_deepspeed_files(mock_venv, capsys):
    """Test when no files contain deepspeed imports."""
    tmp_path, venv_site = mock_venv

    resemble_dir = venv_site / "resemble_enhance"
    resemble_dir.mkdir()
    enhancer_dir = resemble_dir / "enhancer"
    enhancer_dir.mkdir()

    # File without deepspeed
    test_file = resemble_dir / "utils.py"
    test_file.write_text("import torch\ndef helper(): pass\n", encoding="utf-8")

    # __main__.py without torchaudio
    main_py = enhancer_dir / "__main__.py"
    main_py.write_text("def main(): pass\n", encoding="utf-8")

    apply_patches.patch_resemble_enhance()
    captured = capsys.readouterr()
    assert "already applied or not needed" in captured.out


def test_patch_resemble_enhance_torchaudio_not_found(mock_venv, capsys):
    """Test when __main__.py exists but has no torchaudio import."""
    tmp_path, venv_site = mock_venv

    resemble_dir = venv_site / "resemble_enhance"
    resemble_dir.mkdir()
    enhancer_dir = resemble_dir / "enhancer"
    enhancer_dir.mkdir()

    main_py = enhancer_dir / "__main__.py"
    main_py.write_text("import torch\ndef main(): pass\n", encoding="utf-8")

    apply_patches.patch_resemble_enhance()
    captured = capsys.readouterr()
    assert "Could not find 'import torchaudio'" in captured.out


def test_patch_resemble_enhance_read_exception(mock_venv, capsys):
    """Test exception handling during file read."""
    tmp_path, venv_site = mock_venv

    resemble_dir = venv_site / "resemble_enhance"
    resemble_dir.mkdir()

    # Create a directory pretending to be a .py file (will fail read)
    bad_file = resemble_dir / "bad.py"
    bad_file.mkdir()  # Directory, not file

    apply_patches.patch_resemble_enhance()
    capsys.readouterr()
    # Should handle gracefully


# ---------------------------------------------------------
# patch_resemble_cli_args Tests
# ---------------------------------------------------------

def test_patch_resemble_cli_args_no_file(mock_venv, capsys):
    """Test when __main__.py doesn't exist."""
    tmp_path, venv_site = mock_venv

    apply_patches.patch_resemble_cli_args()
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_patch_resemble_cli_args_already_patched(mock_venv, capsys):
    """Test when already has chunk_seconds."""
    tmp_path, venv_site = mock_venv

    enhancer_dir = venv_site / "resemble_enhance" / "enhancer"
    enhancer_dir.mkdir(parents=True)

    main_py = enhancer_dir / "__main__.py"
    main_py.write_text("chunk_seconds=25\ncode", encoding="utf-8")

    apply_patches.patch_resemble_cli_args()
    captured = capsys.readouterr()
    assert "already patched" in captured.out


def test_patch_resemble_cli_args_pattern_not_found(mock_venv, capsys):
    """Test when enhance() pattern is not found."""
    tmp_path, venv_site = mock_venv

    enhancer_dir = venv_site / "resemble_enhance" / "enhancer"
    enhancer_dir.mkdir(parents=True)

    main_py = enhancer_dir / "__main__.py"
    main_py.write_text("def process():\n    pass\n", encoding="utf-8")

    apply_patches.patch_resemble_cli_args()
    captured = capsys.readouterr()
    assert "Could not find" in captured.out


def test_patch_resemble_cli_args_success(mock_venv, capsys):
    """Test successful CLI args patching."""
    tmp_path, venv_site = mock_venv

    enhancer_dir = venv_site / "resemble_enhance" / "enhancer"
    enhancer_dir.mkdir(parents=True)

    main_py = enhancer_dir / "__main__.py"
    content = """
def process():
    hwav, sr = enhance(
        input_file,
        lambd=args.lambd,
        tau=args.tau,
        run_dir=run_dir,
    )
"""
    main_py.write_text(content, encoding="utf-8")

    apply_patches.patch_resemble_cli_args()

    patched = main_py.read_text(encoding="utf-8")
    assert "chunk_seconds" in patched

    captured = capsys.readouterr()
    assert "Successfully patched" in captured.out


def test_patch_resemble_cli_args_already_in_regex(mock_venv, capsys):
    """Test when chunk_seconds is already in the enhance call."""
    tmp_path, venv_site = mock_venv

    enhancer_dir = venv_site / "resemble_enhance" / "enhancer"
    enhancer_dir.mkdir(parents=True)

    main_py = enhancer_dir / "__main__.py"
    content = """
def process():
    hwav, sr = enhance(
        input_file,
        lambd=args.lambd,
        tau=args.tau,
        chunk_seconds=25,
    )
"""
    main_py.write_text(content, encoding="utf-8")

    apply_patches.patch_resemble_cli_args()
    captured = capsys.readouterr()
    assert "already patched" in captured.out


# ---------------------------------------------------------
# patch_soundfile_32bit_default Tests
# ---------------------------------------------------------

def test_patch_soundfile_no_file(mock_venv, capsys):
    """Test when soundfile.py doesn't exist."""
    apply_patches.patch_soundfile_32bit_default()
    captured = capsys.readouterr()
    assert "soundfile.py not found" in captured.out


def test_patch_soundfile_needs_full_patch(mock_venv, capsys):
    """Test full soundfile patching with injection."""
    tmp_path, venv_site = mock_venv

    sf_py = venv_site / "soundfile.py"
    sf_py.write_text("""
def read(file):
    pass

def write(file, data, samplerate, subtype=None, endian=None, format=None,
          closefd=True, compression_level=None, bitrate_mode=None):
    # Write audio
    pass
""", encoding="utf-8")

    apply_patches.patch_soundfile_32bit_default()

    patched = sf_py.read_text(encoding="utf-8")
    assert "subtype='FLOAT'" in patched
    assert "Forced 32-bit" in patched

    captured = capsys.readouterr()
    assert "Updated" in captured.out or "Injected" in captured.out


def test_patch_soundfile_already_patched(mock_venv, capsys):
    """Test when soundfile already has 32-bit patch."""
    tmp_path, venv_site = mock_venv

    sf_py = venv_site / "soundfile.py"
    sf_py.write_text("""
def write(file, data, samplerate, subtype='FLOAT'):
    if subtype in ['PCM_16', 'PCM_24'] or subtype is None: subtype = 'FLOAT'  # Forced 32-bit
    pass
""", encoding="utf-8")

    apply_patches.patch_soundfile_32bit_default()
    captured = capsys.readouterr()
    assert "already present" in captured.out


def test_patch_soundfile_no_def_write(mock_venv, capsys):
    """Test when def write line is not found."""
    tmp_path, venv_site = mock_venv

    sf_py = venv_site / "soundfile.py"
    sf_py.write_text("def read(): pass\n", encoding="utf-8")

    apply_patches.patch_soundfile_32bit_default()
    captured = capsys.readouterr()
    assert "Could not find" in captured.out


# ---------------------------------------------------------
# patch_common_separator_force_soundfile Tests
# ---------------------------------------------------------

def test_patch_separator_no_file(mock_venv, capsys):
    """Test when common_separator.py doesn't exist."""
    apply_patches.patch_common_separator_force_soundfile()
    captured = capsys.readouterr()
    assert "common_separator.py not found" in captured.out


def test_patch_separator_needs_patching(mock_venv, capsys):
    """Test patching common_separator to force soundfile."""
    tmp_path, venv_site = mock_venv

    sep_dir = venv_site / "audio_separator" / "separator"
    sep_dir.mkdir(parents=True)

    sep_py = sep_dir / "common_separator.py"
    sep_py.write_text('''
class Separator:
    def __init__(self, config):
        self.use_soundfile = config.get("use_soundfile")
''', encoding="utf-8")

    apply_patches.patch_common_separator_force_soundfile()

    patched = sep_py.read_text(encoding="utf-8")
    assert "True # Forced" in patched

    captured = capsys.readouterr()
    assert "Successfully forced" in captured.out


def test_patch_separator_already_forced(mock_venv, capsys):
    """Test when already forced."""
    tmp_path, venv_site = mock_venv

    sep_dir = venv_site / "audio_separator" / "separator"
    sep_dir.mkdir(parents=True)

    sep_py = sep_dir / "common_separator.py"
    sep_py.write_text('self.use_soundfile = True # Forced by apply_patches.py', encoding="utf-8")

    apply_patches.patch_common_separator_force_soundfile()
    captured = capsys.readouterr()
    assert "already forced" in captured.out


def test_patch_separator_target_not_found(mock_venv, capsys):
    """Test when target line not found."""
    tmp_path, venv_site = mock_venv

    sep_dir = venv_site / "audio_separator" / "separator"
    sep_dir.mkdir(parents=True)

    sep_py = sep_dir / "common_separator.py"
    sep_py.write_text("class Separator: pass\n", encoding="utf-8")

    apply_patches.patch_common_separator_force_soundfile()
    captured = capsys.readouterr()
    assert "Could not find target" in captured.out


# ---------------------------------------------------------
# Module Main Block Test
# ---------------------------------------------------------

def test_module_import():
    """Test that module imports cleanly."""
    import importlib
    importlib.reload(apply_patches)
    assert hasattr(apply_patches, 'patch_resemble_enhance')
    assert hasattr(apply_patches, 'patch_resemble_cli_args')
    assert hasattr(apply_patches, 'patch_soundfile_32bit_default')
    assert hasattr(apply_patches, 'patch_common_separator_force_soundfile')
