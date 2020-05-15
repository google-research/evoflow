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


def test_prod(backends):
    inputs = [1, 2, 3, 4]
    for B in backends:
        tensor = B.tensor(inputs)
        assert B.prod(tensor) == 24


def test_sum(backends):
    inputs = [1, 2, 3, 4]
    for B in backends:
        tensor = B.tensor(inputs)
        assert B.sum(tensor) == 10


def test_max(backends):
    inputs = [1, 2, 3, 4]
    for B in backends:
        tensor = B.tensor(inputs)
        assert B.max(tensor) == 4


def test_min(backends):
    inputs = [1, 2, 3]
    for B in backends:
        tensor = B.tensor(inputs)
        assert B.min(tensor) == 1


def test_mean(backends):
    inputs = [1, 2, 3]
    for B in backends:
        tensor = B.tensor(inputs)
        assert B.mean(tensor) == 2
