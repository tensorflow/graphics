import tensorflow as tf

use_gpu = tf.test.is_gpu_available(
    cuda_only=True,
    min_cuda_compute_capability=None
)
cpu_device_id = 0
gpu_device_id = 0

def get_device_name():
    """
        Get the current tensorflow device name we are using.
    """
    global use_gpu
    global cpu_device_id
    global gpu_device_id
    return '/device:gpu:' + str(gpu_device_id) if use_gpu else '/device:cpu:' + str(cpu_device_id)

def set_use_gpu(v: bool):
    """
        Set whether to use CUDA or not.
    """
    global use_gpu
    use_gpu = v

def get_use_gpu():
    """
        Get whether we are using CUDA or not.
    """
    global use_gpu
    return use_gpu

def set_cpu_device_id(did: int):
    """
        Set the cpu device id we are using.
    """
    global cpu_device_id
    cpu_device_id = did

def get_cpu_device_id():
    """
        Get the cpu device id we are using.
    """
    global cpu_device_id
    return cpu_device_id

def set_gpu_device_id(did: int):
    """
        Set the gpu device id we are using.
    """
    global gpu_device_id
    gpu_device_id = did

def get_gpu_device_id():
    """
        Get the gpu device id we are using.
    """
    global gpu_device_id
    return gpu_device_id
