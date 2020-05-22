import platform
import psutil
from distutils import spawn
import tensorflow as tf
from subprocess import Popen, PIPE


def get_system_info(gpu_enable):
    """Return system information

    Args:
        gpu_enable (Bool): Is this a GPU test?

    Returns:
        dict: system information.
    """
    info = {}

    # os
    info["os"] = {"name": platform.system(), "version": platform.version()}

    # memory
    mem_info = psutil.virtual_memory()
    info['memory'] = {
        "unit": "MB",
        "used": int(mem_info.used / (1024 * 1024.0)),
        "total": int(mem_info.total / (1024 * 1024.0))
    }

    # cpu
    info['cpu'] = {
        "name": platform.processor(),
        "cores": psutil.cpu_count(),
        "freqency": psutil.cpu_freq().max
    }

    import numpy
    info['python'] = {"numpy": numpy.__version__}
    # tensorflow
    info['python']['tensorflow'] = {
        "visible_gpu": [g.name for g in tf.config.get_visible_devices('GPU')],
        "version": tf.__dict__.get('__version__')
    }

    try:
        import cupy as cp
    except ImportError:
        info['python']['cupy'] = None
    finally:
        info['python']['cupy'] = cp.__version__

    if gpu_enable:
        info['gpu'] = _get_gpu_usage()
        print(tf.config.get_visible_devices())
    else:
        info['gpu'] = []

    return info


def _get_gpu_usage():
    """gpu usage"""
    nvidia_smi = _find_nvidia_smi()

    metrics = {
        "index": "index",
        "utilization.gpu": "usage",
        "memory.used": "used",
        "memory.total": "total",
        "driver_version": "driver",
        # "cuda_version": "cuda", # doesn't exist
        "name": "name",
        "temperature.gpu": "value",
    }
    metrics_list = sorted(metrics.keys())  # deterministic ordered list
    query = ','.join(metrics_list)
    try:
        p = Popen([
            nvidia_smi,
            "--query-gpu=%s" % query, "--format=csv,noheader,nounits"
        ],
                  stdout=PIPE)
        stdout, _ = p.communicate()
    except:  # noqa
        return []

    info = stdout.decode('UTF-8')
    gpus = []
    for l in info.split('\n'):
        if ',' not in l:
            continue
        info = l.strip().split(',')
        gpu_info = {"memory": {"unit": "MB"}, 'temperature': {"unit": 'C'}}
        for idx, metric in enumerate(metrics_list):
            value = info[idx].strip()
            metric_name = metrics[metric]
            if "memory" in metric:
                gpu_info['memory'][metric_name] = int(value)
            # elif "temperature" in metric:
            #    gpu_info['temperature'][metric_name] = int(value)
            elif "driver" in metric:
                gpu_info['driver'] = value
            else:
                gpu_info[metric_name] = value
        gpus.append(gpu_info)
        print(gpus)
    return gpus


def _find_nvidia_smi():
    """
    Find nvidia-smi program used to query the gpu

    Returns:
        str: nvidia-smi path or none if not found
    """

    if platform.system() == "Windows":
        # If the platform is Windows and nvidia-smi
        # could not be found from the environment path,
        # try to find it from system drive with default installation path
        nvidia_smi = spawn.find_executable('nvidia-smi')
        if nvidia_smi is None:
            nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ[  # noqa
                'systemdrive']  # noqa
    else:
        nvidia_smi = "nvidia-smi"
    return nvidia_smi
