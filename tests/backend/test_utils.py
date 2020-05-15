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

from evoflow.backend.common import floatx, intx


def test_to_numpy(backends):
    inputs = [[1, 2, 3], [0.1, 0.2, 0.3], [1, 0.1, 0.2]]
    for ipt in inputs:
        for B in backends:
            tensor = B.tensor(ipt)
            numpy_arr = B.as_numpy_array(tensor)

            # check conversion type is consistent
            assert B.dtype(tensor) == numpy_arr.dtype.name

            # checking that converting back and forth works
            tensor_back = B.tensor(numpy_arr)
            assert B.tensor_equal(tensor, tensor_back)


def test_cast_dtype(backends):
    inputs = [1, 2, 3]
    for B in backends:
        tensor = B.tensor(inputs)
        tensor = B.cast(tensor, floatx())
        assert B.dtype(tensor) == floatx()
        tensor = B.cast(tensor, intx())
        assert B.dtype(tensor) == intx()


def test_reshape(backends):
    inputs = [[1, 2, 3], [4, 5, 6]]
    for B in backends:
        tensor = B.tensor(inputs)
        reshaped_tensor = B.reshape(tensor, 6)
        assert reshaped_tensor.shape == (6, )

        # check back and forth
        reshaped_tensor = B.reshape(reshaped_tensor, (2, 3))
        assert B.tensor_equal(tensor, reshaped_tensor)


def test_is_tensor(backends):
    inputs = [1, 2, 3]
    for B in backends:
        tensor = B.tensor(inputs)
        assert B.is_tensor(tensor)
        assert not B.is_tensor(inputs)


def test_tensor_equal(backends):
    inputs = [1, 2, 3]
    same_inputs = [1, 2, 3]
    different_inputs = [1, 3, 2]

    for B in backends:
        tensor = B.tensor(inputs)
        same_tensor = B.tensor(same_inputs)
        different_tensor = B.tensor(different_inputs)
        assert B.tensor_equal(tensor, same_tensor)
        assert not B.tensor_equal(tensor, different_tensor)


def test_tensor_are_near(backends):
    inputs = [1, 2, 3]
    near_inputs = [1.1, 2.1, 3.1]
    different_inputs = [1, 3, 2]

    for B in backends:
        tensor = B.tensor(inputs)
        tensor = B.cast(tensor, floatx())
        near_tensor = B.tensor(near_inputs)
        different_tensor = B.tensor(different_inputs)
        different_tensor = B.cast(different_tensor, floatx())

        assert B.assert_near(tensor, near_tensor, absolute_tolerance=0.2)

        # those should fail
        assert not B.tensor_equal(tensor, near_tensor)
        assert not B.tensor_equal(tensor, different_tensor)