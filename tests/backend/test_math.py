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


def test_dot(backends):
    inputs = [1, 2, 3]
    inputs2 = [3, 2, 3]

    for B in backends:
        tensor = B.tensor(inputs)
        tensor2 = B.tensor(inputs2)
        product = B.dot(tensor, tensor2)
        assert product == 16


def test_dot_scalar(backends):
    inputs = [1, 2, 3]
    scalar = 2

    for B in backends:
        scalar_t = B.tensor(scalar)
        tensor = B.tensor(inputs)
        product = B.dot(tensor, scalar_t)
        assert B.tensor_equal(product, B.tensor([2, 4, 6]))


def test_dot_scalar_scalar(backends):
    scalar = 2
    scalar2 = 3

    for B in backends:
        t = B.tensor(scalar)
        t2 = B.tensor(scalar2)
        product = B.dot(t, t2)
        assert B.tensor_equal(product, B.tensor(6))


def test_dot_2d_2d(backends):
    arr = [[1, 2, 3], [3, 2, 1]]
    for B in backends:
        t = B.tensor(arr)
        t2 = B.transpose(t)
        product = B.dot(t, t2)
        assert B.tensor_equal(product, B.tensor([[14, 10], [10, 14]]))


def test_dot_2d_1d(backends):
    arr = [[1, 0, 1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1, 1],
           [1, 0, 1, 1, 0, 1, 1, 1]]
    arr2 = [1, 0, 1, 1, 0, 1, 1, 1]
    for B in backends:
        t = B.tensor(arr)
        t2 = B.tensor(arr2)
        product = B.dot(t, t2)
        print('product', product)
        expected = B.tensor([6, 5, 6])
        print('expected', expected)
        assert B.tensor_equal(product, expected)


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
        div_results = B.divide(tensor, tensor2)
        print('result', div_results)
        expected_tensor = B.tensor(expected)
        print('expected', expected_tensor)
        assert B.tensor_equal(div_results, expected_tensor)


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
        norm = B.norm(B.cast(tensor, 'float32'))
        assert norm - expected < 0.0001
