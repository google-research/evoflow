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

import evoflow.backend as B
from copy import copy
from evoflow.ops import UniformCrossover1D, UniformCrossover2D
from evoflow.ops import UniformCrossover3D


def test_uniform_2Dcrossover_randomness_shape():
    GENOME_SHAPE = (10, 4, 4)
    population = B.randint(0, 1024, GENOME_SHAPE)
    population_fraction = 0.5
    crossover_probability = (0.5, 0.5)

    original_population = copy(population)
    OP = UniformCrossover2D(population_fraction, crossover_probability)
    population = OP(population)

    diff = B.clip(abs(population - original_population), 0, 1)
    print(diff)
    expected_mutations = original_population.shape[0] * population_fraction

    mutated_chromosomes = []
    for c in diff:
        if B.max(c):
            mutated_chromosomes.append(c)
    num_mutations = len(mutated_chromosomes)

    # sometime we have a collision so we use a delta
    assert abs(num_mutations - expected_mutations) < 2

    mutated_rows = crossover_probability[0] * GENOME_SHAPE[1]
    mutated_cells = crossover_probability[0] * GENOME_SHAPE[2]
    for cidx, c in enumerate(mutated_chromosomes):
        mr = 0.0
        mc = 0.0
        for r in c:
            s = B.cast(B.sum(r), B.floatx())
            if s:
                mr += 1
                mc += s

        assert abs(mutated_rows - mr) < 2
        assert abs(B.cast(mutated_cells, B.floatx()) -
                   (mc / mutated_rows)) < 2.0  # noqa


def test_uniformcrossover2d_distribution():
    """check that every gene of the tensor are going to be flipped equally
    Note:
    # ! We need enough iterations and chromosomes to reduce collision
    # ! and ensure numerical stability
    """
    NUM_ITERATIONS = 1000
    GENOME_SHAPE = (20, 4, 4)
    population = B.randint(0, 1024, GENOME_SHAPE)
    population_fraction = 1
    crossover_probability = (0.5, 0.5)

    # each gene proba of being mutated 0.5*0.5 > 0.25
    # each chromosome proba of being mutated 1
    # => gene average hit rate: 1000 / (1/4)  ~250
    MIN_DIFF_BOUND = 180
    MAX_DIFF_BOUND = 320

    OP = UniformCrossover2D(population_fraction, crossover_probability)

    # diff matrix
    previous_population = copy(population)
    population = OP(population)
    diff = B.clip(abs(population - previous_population), 0, 1)
    for _ in range(NUM_ITERATIONS - 1):
        previous_population = copy(population)
        population = OP(population)

        curr_diff = B.clip(abs(population - previous_population), 0, 1)
        # acumulating diff matrix
        diff += curr_diff

    print(curr_diff)

    for c in diff:
        print(c)
        print('mean', B.mean(c), 'min', B.min(c), 'max', B.max(c))
        assert B.min(c) > MIN_DIFF_BOUND
        assert B.max(c) < MAX_DIFF_BOUND
        assert MIN_DIFF_BOUND < B.mean(c) < MAX_DIFF_BOUND


def test_ND():
    "test various tensor size random"

    TEST_INPUTS = [
        [UniformCrossover1D, (2, 4), 0.5],
        [UniformCrossover2D, (2, 4, 4), (0.5, 0.5)],
        [UniformCrossover3D, (2, 4, 4, 4), (0.5, 0.5, 0.5)],
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
