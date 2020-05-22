from termcolor import cprint
from .system import get_system_info


def setup(backend):
    # disable gpu if needed - must be as early as possible.
    if backend == 'tensorflow-cpu':
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        cprint('Tensoflow set to CPU', 'green')

    # setting backend
    from evoflow.config import set_backend

    if backend in ['tensorflow-cpu', 'tensorflow-gpu']:
        set_backend('tensorflow')
    else:
        set_backend(backend)
    cprint('Requested backend: %s' % backend, 'magenta')

    if backend in ['tensorflow-gpu', 'cupy']:
        gpu_enable = True
    else:
        gpu_enable = False

    sys_info = get_system_info(gpu_enable)
    return sys_info
