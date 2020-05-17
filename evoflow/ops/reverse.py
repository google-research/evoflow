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
from evoflow.engine import OP
from evoflow import backend as B


class Reverse(OP):
    def __init__(self, population_fraction, max_reverse_probability, **kwargs):
        """Reverse part of the genes.

        Args:
            population_fraction (float): How many chromosomes
            should have some of their genes reversed.

            max_reverse_probability (list(float)): What is the maximum fraction
            of the chromosome that should be affected by the reverse. Number is
            drawon from [0, max_reverse_probability]

            debug (bool, optional): print debug information and function
            returns additional data.

        Returns:
            tensor: population with some genes reversed.

        See:
            https://arxiv.org/pdf/1901.05737.pdf
        """

        if not (0 < population_fraction <= 1.0):
            raise ValueError("population_fraction must be in (0. 1]")

        for val in max_reverse_probability:
            if not (0 < val <= 1.0):
                raise ValueError(
                    "max_reverse_probability values must be between (0. 1]")

        self.population_fraction = population_fraction
        self.max_reverse_probability = max_reverse_probability
        self.has_cache = False
        self.mutation_shape = []
        self.slices = None
        super(Reverse, self).__init__(**kwargs)

    def call(self, populations):
        results = []
        for population in populations:
            results.append(self.compute(population))
        return results

    def compute(self, population):

        if not self.debug:
            population = B.shuffle(population)

        # how many chromosomes to crossover
        num_reversed_chromosomes = int(population.shape[0] *
                                       self.population_fraction)

        self.print_debug('num chromosomes', num_reversed_chromosomes)

        # compute the shape needed for the mutation
        mutations_shape = [num_reversed_chromosomes]
        for idx, frac in enumerate(self.max_reverse_probability):
            max_genes = int(population.shape[idx + 1] * frac + 1)
            num_genes = B.randint(0, high=max_genes)
            mutations_shape.append(num_genes)
            self.print_debug(idx, 'num_genes', num_genes, 'max', max_genes)

        self.mutations_shape = mutations_shape
        self.print_debug("population_shape:", population.shape)
        self.print_debug("mutation_shape:", self.mutations_shape)

        # compute the fancy indexing dynamlically
        # ! the start point must be randomized
        slices = [slice(0, num_reversed_chromosomes)]
        for idx, crossover_size in enumerate(self.mutations_shape[1:]):
            # ! making indexing explicit as its a huge pitfall
            mutation_dim = idx + 1
            max_start = population.shape[mutation_dim] - crossover_size + 1
            start = B.randint(0, max_start, 1)[0]
            slices.append(slice(start, crossover_size + start))

        slices = tuple(slices)
        self.print_debug(slices)

        # revesing
        reversed_population = population[slices]
        axis = B.tensor([x for x in range(1, len(reversed_population.shape))])
        reversed_population = B.reverse(reversed_population, axis)
        self.print_debug('reversed population', reversed_population)

        population = B.assign(population, reversed_population, slices)
        return population


class Reverse1D(Reverse):
    def __init__(self,
                 population_fraction=0.9,
                 max_reverse_probability=0.2,
                 **kwargs):
        if not isinstance(max_reverse_probability, float):
            raise ValueError('max_reverse_probability must be a float')

        super(Reverse1D,
              self).__init__(population_fraction=population_fraction,
                             max_reverse_probability=[max_reverse_probability],
                             **kwargs)


class Reverse2D(Reverse):
    def __init__(self,
                 population_fraction=0.9,
                 max_reverse_probability=(0.2, 0.2),
                 **kwargs):

        if len(max_reverse_probability) != 2:
            raise ValueError('max_reverse_probability must be of form (x, y)')
        super(Reverse2D,
              self).__init__(population_fraction=population_fraction,
                             max_reverse_probability=max_reverse_probability,
                             **kwargs)


class Reverse3D(Reverse):
    def __init__(self,
                 population_fraction=0.9,
                 max_reverse_probability=(0.2, 0.2, 0.2),
                 **kwargs):

        if len(max_reverse_probability) != 3:
            raise ValueError(
                'max_reverse_probability must be of form (x, y, z)')

        super(Reverse3D,
              self).__init__(population_fraction=population_fraction,
                             max_reverse_probability=max_reverse_probability,
                             **kwargs)


if __name__ == '__main__':
    from copy import copy
    GENOME_SHAPE = (6, 4, 4)
    population = B.randint(0, 256, GENOME_SHAPE)
    population_fraction = 0.5
    max_reverse_probability = (0.5, 0.5)
    cprint(population[0], 'blue')
    original_population = copy(population)
    population = Reverse2D(population_fraction,
                           max_reverse_probability,
                           debug=True)(population)

    cprint(population[0], 'yellow')
    # diff matrix
    diff = B.clip(abs(population - original_population), 0, 1)
    print(diff)
