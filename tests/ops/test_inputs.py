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

from geneflow.ops import RandomInputs, Inputs
from geneflow import backend as B


def test_inputs():
    shape = (128, 64)
    vals = B.randint(1, 10, shape=shape)
    inputs = Inputs(shape)
    inputs.assign(vals)
    assert inputs.call().all() == vals.all()


def test_random_inputs_int():
    shape = (64, 128)
    min_val = 10
    max_val = 20

    inputs = RandomInputs(shape, min_value=min_val,
                          max_value=max_val)

    # NOTE: Inputs OP are not eager so we need to call them explictly
    chromosomes = inputs.call()
    assert chromosomes.shape == shape
    assert B.max(chromosomes) <= max_val
    assert B.min(chromosomes) >= min_val

    chromosomes2 = inputs.call()
    assert chromosomes2.all() == chromosomes.all()


def test_random_inputs_regenerate():
    shape = (64, 128)
    min_val = 10
    max_val = 20

    inputs = RandomInputs(shape, min_value=min_val,
                          max_value=max_val)
    # NOTE: Inputs OP are not eager so we need to call them explictly
    chromosomes = inputs.call()
    chromosomes2 = inputs.call(regenerate=True)
    assert B.abs(B.sum(chromosomes2 - chromosomes)) > 0


def test_random_inputs_zero_and_one():
    num_tries = 10
    shape = (64, 128)
    min_val = 0
    max_val = 1

    for _ in range(num_tries):
        inputs = RandomInputs(shape, min_value=min_val,
                              max_value=max_val)
        # NOTE: Inputs OP are not eager so we need to call them explictly
        chromosomes = inputs.call()
        assert B.max(chromosomes, axis=0).all() == 1
        assert B.min(chromosomes, axis=0).all() == 0


def test_random_inputs_zero_and_one_no_minval():
    num_tries = 10
    shape = (64, 128)
    max_val = 1

    for _ in range(num_tries):
        inputs = RandomInputs(shape, max_value=max_val)
        # NOTE: Inputs OP are not eager so we need to call them explictly
        chromosomes = inputs.call()
        assert B.max(chromosomes, axis=0).all() == 1
        assert B.min(chromosomes, axis=0).all() == 0
