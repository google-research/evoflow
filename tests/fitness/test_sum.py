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

from evoflow.fitness import Sum
import evoflow.backend as B


def test_sum2d():
    t = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
         [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
    inputs = B.tensor(t)
    print(inputs)
    result = Sum().call(inputs)
    assert result.shape == (3, )
    print(result)

    expected = B.tensor([9, 9, 9])
    assert B.tensor_equal(result, expected)


def test_max_gene_val_2d():
    MAX_VAL = 10
    t = B.randint(0, MAX_VAL + 1, (10, 10, 10))
    max_sum_value = MAX_VAL * 10 * 10
    v = Sum(max_sum_value=max_sum_value).call(t)
    assert v.shape == (10, )
    for t in v:
        assert t < 1
