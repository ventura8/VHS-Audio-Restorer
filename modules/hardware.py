import os
import subprocess
import torch
import sys
import platform
from pathlib import Path


def get_optimal_settings():
    """Auto-detect hardware and return optimal settings."""
    settings = {
        "cpu_threads": os.cpu_count() or 16,
        "gpu_batch_size": 1,
        "cuda_device": "cuda:0",
        "gpu_vram_gb": 0,
        "profile_name": "Low (Entry Config)"
    }

    # Detect GPU VRAM and calculate optimal batch size
    if torch.cuda.is_available():
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024 ** 3)
            settings["gpu_vram_gb"] = vram_gb

            if vram_gb >= 24:
                settings["gpu_batch_size"] = 32
                settings["profile_name"] = "EXTREME (RTX 5090 / A6000)"
            elif vram_gb >= 22:
                settings["gpu_batch_size"] = 16
                settings["profile_name"] = "ULTRA (RTX 3090/4090)"
            elif vram_gb >= 15:
                settings["gpu_batch_size"] = 8
                settings["profile_name"] = "HIGH (RTX 4080/5080)"
            elif vram_gb >= 10:
                settings["gpu_batch_size"] = 4
                settings["profile_name"] = "MID (RTX 3080/4070)"
            else:
                settings["gpu_batch_size"] = 1
                settings["profile_name"] = "LOW (Entry Config)"

        except Exception:
            pass  # Keep defaults

    return settings


# Auto-configure on import
_hw_settings = get_optimal_settings()

CPU_THREADS = _hw_settings["cpu_threads"]
GPU_BATCH_SIZE = _hw_settings["gpu_batch_size"]
CUDA_DEVICE = _hw_settings["cuda_device"]
GPU_VRAM_GB = _hw_settings["gpu_vram_gb"]
PROFILE_NAME = _hw_settings["profile_name"]


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
    try:
        output = subprocess.check_output("nvidia-smi -L", shell=True).decode()
        # Format: GPU 0: NVIDIA GeForce RTX 5090 (UUID: ...)
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
            nvidia_paths.append(os.path.join(path, "bin"))
            nvidia_paths.append(os.path.join(path, "lib"))
    except ImportError:
        pass

    return nvidia_paths
