import os
import subprocess
import torch
import sys
import platform
from pathlib import Path


def _detect_nvidia_smi(settings):
    """Detects NVIDIA GPU via nvidia-smi."""
    try:
        output = subprocess.check_output("nvidia-smi -L", shell=True, stderr=subprocess.DEVNULL).decode()
        if "NVIDIA" in output:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            settings["is_nvidia"] = True
    except Exception:
        pass


def _detect_pytorch_cuda(settings):
    """Detects GPU via PyTorch and sets VRAM/Profile."""
    if not torch.cuda.is_available():
        settings["device_index"] = 0
        return

    try:
        device_id = 0
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i).upper()
            if "NVIDIA" in name or "RTX" in name or "GTX" in name or "QUADRO" in name:
                device_id = i
                settings["is_nvidia"] = True
                break

        settings["device_index"] = device_id
        settings["cuda_device"] = f"cuda:{device_id}"
        gpu_props = torch.cuda.get_device_properties(device_id)
        vram_gb = gpu_props.total_memory / (1024 ** 3)
        settings["gpu_vram_gb"] = vram_gb

        if vram_gb >= 24:
            settings["gpu_batch_size"] = 32
            settings["profile_name"] = "EXTREME (RTX 3090/4090/5090)"
        elif vram_gb >= 15:
            settings["gpu_batch_size"] = 8
            settings["profile_name"] = "HIGH (RTX 3080/4080/5080)"
        elif vram_gb >= 10:
            settings["gpu_batch_size"] = 4
            settings["profile_name"] = "MID (RTX 3070/4070)"
        else:
            settings["gpu_batch_size"] = 1
            settings["profile_name"] = "LOW (Entry Config)"

    except Exception:
        pass


def _apply_env_vars(settings):
    """Applies environment variables based on settings."""
    if settings["is_nvidia"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(settings["device_index"])
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
        settings["cuda_device"] = "cuda:0"


def get_optimal_settings():
    """Auto-detect hardware and return optimal settings, prioritizing NVIDIA."""
    settings = {
        "cpu_threads": os.cpu_count() or 16,
        "gpu_batch_size": 1,
        "cuda_device": "cuda:0",
        "gpu_vram_gb": 0,
        "profile_name": "Low (Entry Config)",
        "is_nvidia": False,
        "device_index": 0
    }

    _detect_nvidia_smi(settings)
    _detect_pytorch_cuda(settings)
    _apply_env_vars(settings)

    return settings


# Auto-configure on import
_hw_settings = get_optimal_settings()

CPU_THREADS = _hw_settings["cpu_threads"]
GPU_BATCH_SIZE = _hw_settings["gpu_batch_size"]
CUDA_DEVICE = _hw_settings["cuda_device"]
GPU_VRAM_GB = _hw_settings["gpu_vram_gb"]
PROFILE_NAME = _hw_settings["profile_name"]
IS_NVIDIA = _hw_settings["is_nvidia"]
DEVICE_INDEX = _hw_settings["device_index"]


def get_cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
            processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            return processor_name.strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def get_gpu_name():
    # 1. Try PyTorch first (more accurate for what its using)
    if torch.cuda.is_available():
        try:
            # We use the device specified in CUDA_DEVICE
            idx = int(CUDA_DEVICE.split(':')[-1])
            return torch.cuda.get_device_name(idx)
        except Exception:
            pass

    # 2. Fallback to nvidia-smi
    try:
        output = subprocess.check_output("nvidia-smi -L", shell=True, stderr=subprocess.DEVNULL).decode()
        if "NVIDIA" in output:
            return output.split(':')[1].split('(')[0].strip()
    except Exception:
        pass

    return "Generic / Not Detected"


def get_nvidia_paths():
    """Returns a list of paths containing CUDNN/CUBLAS DLLs."""
    nvidia_paths = []

    # Try torch.lib first
    try:
        import torch
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists():
            nvidia_paths.append(str(torch_lib))
    except ImportError:
        pass

    # Try nvidia.* packages
    try:
        import nvidia.cudnn  # type: ignore
        import nvidia.cublas  # type: ignore
        for lib in [nvidia.cudnn, nvidia.cublas]:
            if hasattr(lib, '__path__') and lib.__path__:
                path = lib.__path__[0]
            else:
                path = os.path.dirname(lib.__file__)
            
            bin_path = os.path.join(path, "bin")
            lib_path = os.path.join(path, "lib")
            if os.path.exists(bin_path):
                nvidia_paths.append(bin_path)
            if os.path.exists(lib_path):
                nvidia_paths.append(lib_path)
    except ImportError:
        pass

    return nvidia_paths
