import torch

try:
    from pynvml import *
except ImportError:
    try:
        import nvidia_smi as nvml
        # nvidia-ml-py uses pynvml under the hood, same API
        from pynvml import *
    except ImportError:
        print("Warning: nvidia-ml-py not installed. GPU stats will not be available.")
        nvmlInit = lambda: None

try:
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
except Exception as e:
    print(f"Warning: Could not initialize NVML: {e}")
    h = None

def gpu_stats():
    if h is None:
        print("GPU stats not available")
        return
    try:
        mem = nvmlDeviceGetMemoryInfo(h)
        util = nvmlDeviceGetUtilizationRates(h)
        print(f"GPU Util: {util.gpu}% | Mem: {mem.used/1e9:.2f}/{mem.total/1e9:.2f} GB")
    except Exception as e:
        print(f"Error getting GPU stats: {e}")

def estimate_vram(cfg):
    total = cfg["hidden_size"]**2 * cfg["n_layers"]*3*2/1e9 + 0.75
    print(f"Estimated VRAM needed: {total:.2f} GB")
