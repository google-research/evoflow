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
from termcolor import cprint
from copy import copy
from evoflow import backend as B
from evoflow.ops import DualCrossover1D, DualCrossover2D
from evoflow.ops import DualCrossover3D


def test_ND():
    "test various tensor size random"

    TEST_INPUTS = [
        [DualCrossover1D, (2, 4), 0.5],
        [DualCrossover2D, (2, 4, 4), (0.5, 0.5)],
        [DualCrossover3D, (2, 4, 4, 4), (0.5, 0.5, 0.5)],
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


def test_crossover1D_output_shape():
    POPULATION_SHAPE = (8, 6)
    population = B.randint(0, 1024, POPULATION_SHAPE)
    population_fraction = 0.5
    mutations_probability = 0.2

    original_population = copy(population)
    population = DualCrossover1D(population_fraction,
                                 mutations_probability,
                                 debug=True)(population)

    cprint(population, 'cyan')
    cprint(original_population, 'yellow')

    assert population.shape == POPULATION_SHAPE
    # measuring mutation rate
    diff = B.clip(abs(population - original_population), 0, 1)

    # row test
    num_ones_in_row = 0
    for col in diff:
        num_ones_in_row = max(list(col).count(1), num_ones_in_row)

    max_one_in_row = int(POPULATION_SHAPE[1] * mutations_probability)
    assert num_ones_in_row == max_one_in_row
    assert num_ones_in_row


def test_dualcrossover2d_distribution():
    """check that every gene of the tensor are going to be flipped equally
    Note:
    # ! We need enough iterations and chromosomes to reduce collision
    # ! and ensure numerical stability
    """
    NUM_ITERATIONS = 1000
    GENOME_SHAPE = (100, 4, 4)
    population = B.randint(0, 1024, GENOME_SHAPE)
    population_fraction = 1
    crossover_probability = (0.5, 0.5)

    OP = DualCrossover2D(population_fraction, crossover_probability)

    # diff matrix
    previous_population = copy(population)
    population = OP(population)
    diff = B.clip(abs(population - previous_population), 0, 1)
    print(diff)

    for _ in range(NUM_ITERATIONS - 1):
        previous_population = copy(population)
        population = OP(population)

        curr_diff = B.clip(abs(population - previous_population), 0, 1)
        # acumulating diff matrix
        diff += curr_diff

    # print(curr_diff)

    for c in diff:
        print(c)
        print('mean', B.mean(c), 'min', B.min(c), 'max', B.max(c))
        assert B.min(c) > 50
        assert B.max(c) < NUM_ITERATIONS // 2
        assert 200 < B.mean(c) < NUM_ITERATIONS // 2
