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

from evoflow.config import floatx, set_floatx, intx, set_intx


def test_tensor(backends):
    data = [[[1, 2, 3, 4], intx()], [[0.1, 2, 3], floatx()],
            [[3, 0.1, 3], floatx()]]
    for d in data:
        for B in backends:
            tensor = B.tensor(d[0])
            assert B.dtype(tensor) == d[1]


def test_set_floatx_tensor(backends):
    data = [[1.1, 2, 3, 4]]
    for d in data:
        for B in backends:
            set_floatx('float16')
            tensor = B.tensor(d)
            assert B.dtype(tensor) == floatx()
            set_floatx('float32')


def test_set_intx_tensor(backends):
    data = [[1, 2, 3, 4]]
    for d in data:
        for B in backends:
            set_intx('uint8')
            tensor = B.tensor(d)
            assert B.dtype(tensor) == intx()
            set_intx('int32')


def test_set_tensor_dtype(backends):
    data = [[1, 2, 3, 4]]
    for d in data:
        for B in backends:
            tensor = B.tensor(d, 'int16')
            assert B.dtype(tensor) == 'int16'


def test_copy(backends):
    data = [[1, 2, 3, 4]]
    for d in data:
        for B in backends:
            tensor = B.tensor(d)
            tensor2 = B.copy(tensor)

            # same before modification
            assert B.tensor_equal(tensor, tensor2)

            # different after modification
            tensor2 = B.assign(tensor2, 42, slice(2, 3))
            print(tensor2)
            assert not B.tensor_equal(tensor, tensor2)


def test_zero(backends):
    shape = (10, 10)

    for B in backends:
        tensor = B.zeros(shape)
        assert tensor.shape == shape
        assert not B.sum(tensor)


def test_ones(backends):
    shape = (10, 10)

    for B in backends:
        tensor = B.ones(shape)
        assert tensor.shape == shape
        assert B.sum(tensor) == 100


def test_fill(backends):
    shape = (10, 10)

    for B in backends:
        tensor = B.fill(shape, 5)
        assert tensor.shape == shape
        assert B.sum(tensor) == 500
