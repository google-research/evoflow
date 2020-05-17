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


def test_roll(backends):
    inputs = [1, 2, 3]
    expected = [3, 1, 2]

    for B in backends:
        tensor = B.tensor(inputs)
        result = B.roll(tensor, 1, axis=0)
        expected_tensor = B.tensor(expected)
        assert B.tensor_equal(result, expected_tensor)


def test_roll_2d(backends):
    inputs = [[1, 2, 3], [1, 2, 3]]
    expected = [[3, 1, 2], [3, 1, 2]]

    for B in backends:
        tensor = B.tensor(inputs)
        result = B.roll(tensor, 1, axis=1)
        expected_tensor = B.tensor(expected)
        assert B.tensor_equal(result, expected_tensor)


def test_assign(backends):
    inputs = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
    values = [[1, 1], [1, 1]]
    slices = (slice(1, 3), slice(1, 3))

    expected = [[11, 12, 13], [21, 1, 1], [31, 1, 1]]

    for B in backends:
        tval = B.tensor(values)
        tensor = B.tensor(inputs)
        result = B.assign(tensor, tval, slices)
        expected_tensor = B.tensor(expected)
        assert B.tensor_equal(result, expected_tensor)


def test_concatenate(backends):
    inputs = [1, 2, 3]
    inputs2 = [4, 5, 6]
    expected = [1, 2, 3, 4, 5, 6]
    for B in backends:
        tensor = B.tensor(inputs)
        tensor2 = B.tensor(inputs2)
        result = B.concatenate([tensor, tensor2])
        expected_tensor = B.tensor(expected)
        assert B.tensor_equal(result, expected_tensor)
