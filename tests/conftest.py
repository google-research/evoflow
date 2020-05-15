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

import pytest
from termcolor import cprint
# logging
import logging
logger = logging.getLogger()

# need to be done as early as possible
try:
    import tensorflow as tf
except:  # noqa
    cprint('Tensoflow not found', 'yellow')
    pass
else:
    tf.config.set_visible_devices([], 'GPU')
    cprint('Tensoflow set to CPU', 'green')


def pytest_configure(config):
    backend = config.option.backend
    if backend:
        from evoflow.config import set_backend
        set_backend(backend)
        cprint('Requested backend: %s' % backend, 'magenta')
    else:
        cprint("Requested backend: default (tensorflow)", 'magenta')


def pytest_addoption(parser):
    parser.addoption("--backend",
                     help="specify the backend: numpy, cupy, tensorflow")


@pytest.fixture(scope="session")
def backends():
    """constructing a list of backend

    # ! only use this for backend testing. Higher level function import the
    # ! backend so it will create type conflict.
    """
    import evoflow.backend.numpy as NP

    try:
        import cupy as cp  # noqa
        logger.info('cupy found')
    except:  # noqa
        import evoflow.backend.numpy as CP
        logger.info('cupy not found - using numpy instead')
    else:
        import evoflow.backend.cupy as CP

    try:
        import tensorflow  # noqa
        logger.info('TensorFlow found')
    except:  # noqa
        import evoflow.backend.numpy as TF
        logger.info('TensorFlow not found - using numpy instead')
    else:
        import evoflow.backend.tensorflow as TF

    return [NP, CP, TF]
