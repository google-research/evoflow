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

# import os
# os.environ['evoflow_BACKEND'] = 'numpy'

from copy import copy
from termcolor import cprint
from evoflow import backend as B  # noqa: F402
from evoflow.ops import SingleCrossover1D, SingleCrossover2D
from evoflow.ops import SingleCrossover3D


def test_ND():
    "test various tensor size random"

    TEST_INPUTS = [
        [SingleCrossover1D, (2, 4), 0.5],
        [SingleCrossover2D, (2, 4, 4), (0.5, 0.5)],
        [SingleCrossover3D, (2, 4, 4, 4), (0.5, 0.5, 0.5)],
    ]

    for inputs in TEST_INPUTS:
        OP = inputs[0]
        pop_shape = inputs[1]
        mutations_probability = inputs[2]
        population_fraction = 1
        population = B.randint(0, 1024, pop_shape)

        # eager
        RM = OP(population_fraction=population_fraction,
                mutations_probability=mutations_probability)

        population = RM(population)
        assert B.is_tensor(population)
        assert population.shape == pop_shape

        # graph
        RM = OP(population_fraction=population_fraction,
                mutations_probability=mutations_probability)

        population = RM._call_from_graph(population)
        assert B.is_tensor(population)
        assert population.shape == pop_shape


def test_1D_shape():
    POPULATION_SHAPE = (64, 16)
    population = B.randint(0, 1024, POPULATION_SHAPE)
    population_fraction = 0.5
    crossover_size_fraction = 0.2

    original_population = copy(population)
    population = SingleCrossover1D(population_fraction,
                                   crossover_size_fraction,
                                   debug=1)(population)

    cprint(population, 'cyan')
    cprint(original_population, 'yellow')

    assert population.shape == POPULATION_SHAPE
    # measuring mutation rate
    diff = B.clip(abs(population - original_population), 0, 1)
    cprint('diff', 'cyan')
    cprint(diff, 'cyan')

    # row test
    num_ones_in_row = 0
    for col in diff:
        num_ones = list(col).count(1)
        num_ones_in_row = max(num_ones, num_ones_in_row)

    max_one_in_row = POPULATION_SHAPE[1] * crossover_size_fraction
    assert num_ones_in_row <= max_one_in_row
    assert num_ones_in_row

    # col
    diff = B.transpose(diff)
    num_ones_in_col = 0
    for col in diff:
        num_ones_in_col = max(list(col).count(1), num_ones_in_col)

    max_one_in_col = POPULATION_SHAPE[0] * population_fraction
    assert max_one_in_col - 2 <= num_ones_in_col <= max_one_in_col
