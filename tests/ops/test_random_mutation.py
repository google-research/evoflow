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

from evoflow.ops import RandomMutations1D, RandomMutations2D
from evoflow.ops import RandomMutations3D
from evoflow import backend as B
from evoflow.population import randint_population
from copy import copy
from termcolor import cprint


def test_binary_val_default_params():
    pop_shape = (6, 4)
    population = randint_population(pop_shape, 2)
    population = RandomMutations1D(max_gene_value=1, debug=1)(population)
    print(population)
    assert B.max(population) == 1
    assert not B.min(population)


def test_clip():
    pop_shape = (100, 100)
    population = randint_population(pop_shape, 100)
    population = RandomMutations1D(max_gene_value=100, debug=1)(population)
    assert B.max(population) <= 100
    assert not B.min(population)


def test_mutation2d_eager():
    pop_shape = (10, 4, 4)
    max_gene_value = 10
    min_gene_value = 0
    population_fraction = 1
    mutations_probability = (0.5, 0.5)
    min_mutation_value = 1
    max_mutation_value = 1

    population = randint_population(pop_shape, max_gene_value)

    # save original
    original_population = copy(population)
    cprint('[Initial genepool]', 'blue')
    cprint(original_population, 'blue')

    for _ in range(3):
        RM = RandomMutations2D(population_fraction=population_fraction,
                               mutations_probability=mutations_probability,
                               min_gene_value=min_gene_value,
                               max_gene_value=max_gene_value,
                               min_mutation_value=min_mutation_value,
                               max_mutation_value=max_mutation_value,
                               debug=True)
        population = RM(population)

        cprint('\n[Mutated genepool]', 'yellow')
        cprint(population, 'yellow')

        cprint('\n[Diff]', 'magenta')
        diff = population - original_population
        cprint(diff, 'magenta')

        assert B.is_tensor(population)
        assert population.shape == pop_shape
        assert B.max(diff) <= max_mutation_value
        ok = 0
        for chromosome in diff:
            if B.sum(chromosome) >= 2:
                ok += 1

        if (ok / len(diff)) > 0.5:
            break

    assert (ok / len(diff)) > 0.5


def test_mutation2d_graph_mode():
    "make sure the boxing / unboxing works in graph mode"
    pop_shape = (2, 4, 4)
    max_gene_value = 10
    min_gene_value = 0
    population_fraction = 1
    mutations_probability = (0.5, 0.5)
    min_mutation_value = 1
    max_mutation_value = 1

    population = B.randint(0, max_gene_value, pop_shape)

    RM = RandomMutations2D(population_fraction=population_fraction,
                           mutations_probability=mutations_probability,
                           min_gene_value=min_gene_value,
                           max_gene_value=max_gene_value,
                           min_mutation_value=min_mutation_value,
                           max_mutation_value=max_mutation_value,
                           debug=True)

    population = RM._call_from_graph(population)
    assert B.is_tensor(population)
    assert population.shape == pop_shape


def test_ND():
    "test various tensor size random"

    TEST_INPUTS = [
        [RandomMutations1D, (2, 4), 0.5],
        [RandomMutations2D, (2, 4, 4), (0.5, 0.5)],
        [RandomMutations3D, (2, 4, 4, 4), (0.5, 0.5, 0.5)],
    ]

    for inputs in TEST_INPUTS:
        OP = inputs[0]
        pop_shape = inputs[1]
        mutations_probability = inputs[2]

        max_gene_value = 10
        min_gene_value = 0
        population_fraction = 1
        min_mutation_value = 1
        max_mutation_value = 1

        population = randint_population(pop_shape, max_gene_value)

        # eager
        RM = OP(population_fraction=population_fraction,
                mutations_probability=mutations_probability,
                min_gene_value=min_gene_value,
                max_gene_value=max_gene_value,
                min_mutation_value=min_mutation_value,
                max_mutation_value=max_mutation_value)

        population = RM(population)
        assert B.is_tensor(population)
        assert population.shape == pop_shape

        # graph
        RM = OP(population_fraction=population_fraction,
                mutations_probability=mutations_probability,
                min_gene_value=min_gene_value,
                max_gene_value=max_gene_value,
                min_mutation_value=min_mutation_value,
                max_mutation_value=max_mutation_value)

        population = RM._call_from_graph(population)
        assert B.is_tensor(population)
        assert population.shape == pop_shape


def test_uniform_distribution():
    """check that every gene of the tensor are going to be flipped equally
    Note:
    # ! We need enough iterations and chromosomes to reduce collision
    # ! and ensure numerical stability
    """

    NUM_ITERATIONS = 300
    GENOME_SHAPE = (20, 4, 4)
    population = B.randint(0, 1024, GENOME_SHAPE)
    population_fraction = 1
    crossover_probability = (0.5, 0.5)

    OP = RandomMutations2D(population_fraction, crossover_probability)

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
        assert B.min(c) > NUM_ITERATIONS // 15
        assert B.max(c) < NUM_ITERATIONS // 2
        assert NUM_ITERATIONS // 8 < B.mean(c) < NUM_ITERATIONS // 2
