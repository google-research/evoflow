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

def test_add(backends):
    inputs = [1, 2, 3]
    inputs2 = [1, 2, 3]
    expected = [2, 4, 6]
    for B in backends:
        tensor = B.tensor(inputs)
        tensor2 = B.tensor(inputs2)
        assert B.tensor_equal(B.add(tensor, tensor2), B.tensor(expected))


def test_substract(backends):
    inputs = [3, 6, 3]
    inputs2 = [1, 2, 6]
    expected = [2, 4, -3]
    for B in backends:
        tensor = B.tensor(inputs)
        tensor2 = B.tensor(inputs2)
        assert B.tensor_equal(B.subtract(tensor, tensor2), B.tensor(expected))


def test_multiply(backends):
    inputs = [3, 6, 3]
    inputs2 = [1, 2, -6]
    expected = [3, 12, -18]
    for B in backends:
        tensor = B.tensor(inputs)
        tensor2 = B.tensor(inputs2)
        assert B.tensor_equal(B.multiply(tensor, tensor2), B.tensor(expected))


def test_divide(backends):
    inputs = [3, 6, -3]
    inputs2 = [1, 2, 6]
    expected = [3, 3, -0.5]
    for B in backends:
        tensor = B.tensor(inputs)
        tensor2 = B.tensor(inputs2)
        assert B.tensor_equal(B.divide(tensor, tensor2), B.tensor(expected))


def test_mod(backends):
    inputs = [5, 6, -3]
    inputs2 = [3, 2, 6]
    expected = [2, 0, 3]
    for B in backends:
        tensor = B.tensor(inputs)
        tensor2 = B.tensor(inputs2)
        assert B.tensor_equal(B.mod(tensor, tensor2), B.tensor(expected))


def test_norm(backends):
    inputs = [1, 2, 3]
    expected = 3.7416573867739413
    for B in backends:
        tensor = B.tensor(inputs)
        assert B.norm(tensor) == expected
