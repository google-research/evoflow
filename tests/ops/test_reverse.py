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

from termcolor import cprint
from evoflow import backend as B
from evoflow.ops import Reverse1D, Reverse2D, Reverse3D
from evoflow.population import uniform_population


def test_ND():
    "test various tensor size random"

    TEST_INPUTS = [
        [Reverse1D, (2, 4), 0.5],
        [Reverse2D, (2, 4, 4), (0.5, 0.5)],
        [Reverse3D, (2, 4, 4, 4), (0.5, 0.5, 0.5)],
    ]

    for inputs in TEST_INPUTS:
        OP = inputs[0]
        pop_shape = inputs[1]
        max_reverse_probability = inputs[2]
        population_fraction = 1
        population = B.randint(0, 1024, pop_shape)

        # eager
        RM = OP(population_fraction=population_fraction,
                max_reverse_probability=max_reverse_probability)

        population = RM(population)
        assert B.is_tensor(population)
        assert population.shape == pop_shape

        # graph
        RM = OP(population_fraction=population_fraction,
                max_reverse_probability=max_reverse_probability)

        population = RM._call_from_graph(population)
        assert B.is_tensor(population)
        assert population.shape == pop_shape


def test_1D_tensor_values_maintained():
    POPULATION_SHAPE = (6, 8)
    # need a uniform population otherwise its not testable.
    population = uniform_population(POPULATION_SHAPE)
    population_fraction = 0.5
    max_reverse_probability = 0.2

    cprint(population, 'cyan')
    population = Reverse1D(population_fraction,
                           max_reverse_probability,
                           debug=1)(population)

    cprint(population, 'yellow')
    for chrm in population:
        _, _, count = B.unique_with_counts(chrm)
        assert B.max(count) == 1
