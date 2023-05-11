import pynvml

def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB")

    for dev_id in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            print(
                "pid %d using %d MB of memory on device %d."
                % (proc.pid, proc.usedGpuMemory // 1024**2, dev_id)
            )