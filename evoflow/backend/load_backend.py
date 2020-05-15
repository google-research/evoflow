# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from evoflow.config import set_backend, get_backend

# override existing setting to allow reload
if 'GENEFLOW_BACKEND' in os.environ:
    set_backend(os.environ['GENEFLOW_BACKEND'])
elif not get_backend():
    # check if we find cupy
    try:
        import tensorflow  # noqa: F403, F401
    except ImportError:
        set_backend('numpy')
    else:
        set_backend('tensorflow')

if get_backend() == 'numpy':
    from .numpy import *  # noqa: F403, F401
elif get_backend() == 'tensorflow':
    from .tensorflow import *  # noqa: F403, F401
    # ensure we don't run out of memory
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            msg = "GPUs Physical: %d - Logical GPU:%d\n" % (len(gpus),
                                                            len(logical_gpus))
            sys.stderr.write(msg)

elif get_backend() == 'cupy':
    from .cupy import *  # noqa: F403, F401

sys.stderr.write('Using %s backend\n' % get_backend())
