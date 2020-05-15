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

from termcolor import cprint
from evoflow.backend.common import intx


def test_take(backends):
    data = [
        # tensor, indices to take, expected
        [[0, 1, 2, 3, 4, 5], [5, 1, 3], [5, 1, 3]],
    ]
    for d in data:
        for B in backends:
            tensor = B.tensor(d[0])
            idxs = B.tensor(d[1])
            expected = B.tensor(d[2])

            result = B.take(tensor, idxs)
            assert B.tensor_equal(result, expected)


def test_topk(backends):
    data = [
        # tensor, k, expected
        [[1, 10, 2, 20, 3, 30], 3, [5, 3, 1]],
        [[-1, 10, 2, -20, 3, 30], 3, [5, 1, 4]],
    ]
    for d in data:
        for B in backends:
            tensor = B.tensor(d[0])
            k = d[1]
            expected = B.tensor(d[2])

            result = B.top_k_indices(tensor, k)
            print(result)
            assert B.tensor_equal(result, expected)


def test_bottom_k_indices(backends):
    data = [
        # tensor, k, expected
        [[1, 10, 2, 20, 3, 30], 3, [0, 2, 4]],
        [[-1, 10, 2, -20, 3, 30], 3, [3, 0, 2]],
    ]
    for d in data:
        for B in backends:
            cprint('intx dtype:%s' % intx(), 'magenta')
            tensor = B.tensor(d[0])
            cprint('tensor dtype:%s' % B.dtype(tensor), 'green')
            k = d[1]
            expected = B.tensor(d[2])
            cprint('expected dtype:%s' % B.dtype(expected), 'cyan')

            result = B.bottom_k_indices(tensor, k)
            cprint('expected dtype:%s' % B.dtype(result), 'yellow')

            assert B.tensor_equal(result, expected)
