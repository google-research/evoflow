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

if 'GENEFLOW_BACKEND' in os.environ:
    _BACKEND = os.environ['GENEFLOW_BACKEND']
else:
    # check if we find cupy
    try:
        import cupy   # noqa: F403, F401
    except ImportError:
        _BACKEND = 'numpy'
    else:
        _BACKEND = 'cupy'

if _BACKEND == 'numpy':
        from .numpy import *  # noqa: F403, F401
elif _BACKEND == 'cupy':
        from .cupy import *  # noqa: F403, F401
else:
    raise ImportError("Can't find requested backend ", _BACKEND)

sys.stderr.write('Using %s backend\n' % _BACKEND)


def backend():
    "Return the backend used"
    return _BACKEND
