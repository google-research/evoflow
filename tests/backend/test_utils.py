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
