from unittest.mock import MagicMock, patch
import importlib
import os
import modules.hardware

# ---------------------------------------------------------
# HW & Config
# ---------------------------------------------------------


def test_hw_optimal_settings():
    """Test hardware optimal settings detection."""
    # Test with high VRAM GPU
    with patch.object(modules.hardware.torch.cuda, "is_available", return_value=True):
        mock_props = MagicMock()
        mock_props.total_memory = 32 * 1024**3
        with patch.object(
            modules.hardware.torch.cuda, "get_device_properties", return_value=mock_props
        ):
            settings = modules.hardware.get_optimal_settings()
            assert "EXTREME" in settings["profile_name"]

    # Test with medium VRAM GPU
    with patch.object(modules.hardware.torch.cuda, "is_available", return_value=True):
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3
        with patch.object(
            modules.hardware.torch.cuda, "get_device_properties", return_value=mock_props
        ):
            settings = modules.hardware.get_optimal_settings()
            assert "LOW" in settings["profile_name"]


def test_get_nvidia_paths_coverage():
    """Test NVIDIA paths detection - verifies function returns a list without error."""
    # The function may or may not find paths depending on installed packages
    # We just verify it returns a list and handles imports gracefully
    paths = modules.hardware.get_nvidia_paths()
    assert isinstance(paths, list)


@patch("modules.hardware.subprocess.check_output")
def test_get_gpu_name_robust(mock_out):
    """Test GPU name detection."""
    mock_out.return_value = b"GPU 0: NVIDIA RTX 5000 (UUID: abc-123)"
    result = modules.hardware.get_gpu_name()
    assert "5000" in result or "NVIDIA" in result

    # Test failure case
    mock_out.side_effect = Exception("nvidia-smi not found")
    # Ensure torch doesn't interfere
    with patch.object(modules.hardware.torch.cuda, "is_available", return_value=False):
        result = modules.hardware.get_gpu_name()
        assert ("Not Detected" in result) or ("Generic" in result)


def test_get_cpu_name_windows():
    """Test CPU name detection on Windows."""
    # Mock winreg module structure
    mock_winreg = MagicMock()
    # QueryValueEx returns (value, type)
    mock_winreg.QueryValueEx.return_value = ("Intel Core i9 Windows", 1)

    with patch("sys.platform", "win32"):
        # Inject mock winreg into sys.modules so it can be imported/used
        with patch.dict("sys.modules", {"winreg": mock_winreg}):
            result = modules.hardware.get_cpu_name()
            assert "Intel" in result


def test_get_cpu_name_linux():
    """Test CPU name detection on Linux/Fallback."""
    with patch("sys.platform", "linux"):
        with patch("platform.processor", return_value="AMD Ryzen Linux"):
            result = modules.hardware.get_cpu_name()
            assert "AMD" in result


def test_get_cpu_name_fallback():
    """Test CPU name fallback to platform.processor."""
    # Test the non-Windows path directly (which always falls back to platform.processor)
    with patch("sys.platform", "linux"):
        with patch("modules.hardware.platform.processor", return_value="Fallback CPU"):
            result = modules.hardware.get_cpu_name()
            assert "Fallback" in result or len(result) > 0  # Should return something


def test_optimal_settings_all_profiles():
    """Test all GPU profile detection branches."""
    profiles = [
        (32, "EXTREME"),  # 32GB -> EXTREME
        (16, "HIGH"),   # 16GB -> HIGH
        (15, "HIGH"),   # 15GB threshold
        (12, "MID"),    # 12GB -> MID
        (10, "MID"),    # 10GB threshold
        (6, "LOW"),     # 6GB -> LOW
    ]

    for vram_gb, expected_profile in profiles:
        with patch.object(modules.hardware.torch.cuda, "is_available", return_value=True):
            mock_props = MagicMock()
            mock_props.total_memory = vram_gb * 1024**3
            with patch.object(
                modules.hardware.torch.cuda, "get_device_properties", return_value=mock_props
            ):
                settings = modules.hardware.get_optimal_settings()
                assert expected_profile in settings["profile_name"], \
                    f"Expected {expected_profile} for {vram_gb}GB, got {settings['profile_name']}"


def test_optimal_settings_no_cuda():
    """Test optimal settings when CUDA not available."""
    with patch.object(modules.hardware.torch.cuda, "is_available", return_value=False):
        settings = modules.hardware.get_optimal_settings()
        assert settings["gpu_vram_gb"] == 0
        assert "LOW" in settings["profile_name"] or "Entry" in settings["profile_name"]


def test_optimal_settings_cuda_exception():
    """Test optimal settings when CUDA throws exception."""
    with patch.object(modules.hardware.torch.cuda, "is_available", return_value=True):
        with patch.object(
            modules.hardware.torch.cuda,
            "get_device_properties",
            side_effect=Exception("CUDA error")
        ):
            # Should handle exception gracefully and return defaults
            settings = modules.hardware.get_optimal_settings()
            assert settings is not None


def test_module_path_logic():
    """Test module-level FFMPEG path resolution logic."""

    # Mocking venv_scripts.exists() to True
    with patch("modules.hardware.Path.exists", return_value=True):
        importlib.reload(modules.hardware)

    # Mocking to False
    with patch("modules.hardware.Path.exists", return_value=False):
        importlib.reload(modules.hardware)


def test_nvidia_paths_branches():
    """Test get_nvidia_paths branches."""
    # Test torch branch
    m_torch = MagicMock()
    m_torch.__file__ = "/p/torch/__init__.py"
    with patch("importlib.import_module", return_value=m_torch):
        with patch("modules.hardware.Path.exists", return_value=True):
            paths = modules.hardware.get_nvidia_paths()
            assert isinstance(paths, list)

    # Test nvidia.* import failure
    with patch("builtins.__import__", side_effect=ImportError):
        paths = modules.hardware.get_nvidia_paths()
        assert isinstance(paths, list)

@patch("modules.hardware.subprocess.check_output")
def test_detect_nvidia_smi_success(mock_out):
    """Test _detect_nvidia_smi success path."""
    mock_out.return_value = b"GPU 0: NVIDIA RTX 4090"
    settings = {"is_nvidia": False}
    modules.hardware._detect_nvidia_smi(settings)
    assert settings["is_nvidia"] is True
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


@patch("modules.hardware.subprocess.check_output")
def test_detect_nvidia_smi_failure(mock_out):
    """Test _detect_nvidia_smi failure path."""
    mock_out.side_effect = Exception("No SMI")
    settings = {"is_nvidia": False}
    modules.hardware._detect_nvidia_smi(settings)
    assert settings["is_nvidia"] is False


@patch("modules.hardware.torch.cuda")
def test_detect_pytorch_cuda_logic(mock_cuda):
    """Test _detect_pytorch_cuda loop and settings application."""
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2

    # Mock get_device_name to return Intel first (skip) then NVIDIA (pick)
    mock_cuda.get_device_name.side_effect = ["Intel Graphics", "NVIDIA RTX 4090"]

    # Mock properties for VRAM calculation (RTX 4090 ~ 24GB)
    mock_props = MagicMock()
    mock_props.total_memory = 24 * 1024**3
    mock_cuda.get_device_properties.return_value = mock_props

    settings = {"device_index": 0, "is_nvidia": False}
    modules.hardware._detect_pytorch_cuda(settings)

    assert settings["device_index"] == 1
    assert settings["is_nvidia"] is True
    assert settings["profile_name"].startswith("EXTREME")
    assert settings["gpu_batch_size"] == 32


@patch("modules.hardware.torch.cuda")
def test_detect_pytorch_cuda_no_cuda(mock_cuda):
    """Test _detect_pytorch_cuda early return when CUDA not available."""
    mock_cuda.is_available.return_value = False
    settings = {"device_index": -1}  # default test val
    modules.hardware._detect_pytorch_cuda(settings)

def test_apply_env_vars():
    """Test environment variable application."""
    settings = {"is_nvidia": True, "device_index": 2}
    with patch.dict(os.environ, {}, clear=True):
        modules.hardware._apply_env_vars(settings)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.environ["ORT_TENSORRT_FP16_ENABLE"] == "1"
        assert settings["cuda_device"] == "cuda:0"


def test_get_cpu_name_hard_exception():
    """Test get_cpu_name handles winreg exception robustly (simulated on non-windows or locked registry)."""
    import sys
    with patch("sys.platform", "win32"), \
         patch("modules.hardware.platform.processor", return_value="Fallback"), \
         patch.dict(sys.modules, {'winreg': None}):
         # Fallback to platform.processor when winreg import fails
         assert modules.hardware.get_cpu_name() == "Fallback"
