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

from geneflow.ops import RandomMutations1D, RandomMutations2D
from geneflow.ops import RandomMutations3D
from geneflow import backend as B
from copy import copy
from termcolor import cprint


def test_mutation2d_eager():
    pop_shape = (2, 4, 4)
    max_gene_value = 10
    min_gene_value = 0
    population_fraction = 1
    mutations_probability = (0.5, 0.5)
    min_mutation_value = 1
    max_mutation_value = 1

    population = B.randint(0, max_gene_value, pop_shape)

    # save original
    original_population = copy(population)
    cprint('[Initial genepool]', 'blue')
    cprint(original_population, 'blue')

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
    for chromosome in diff:
        assert B.sum(chromosome) == 4


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

        population = B.randint(0, max_gene_value, pop_shape)

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
