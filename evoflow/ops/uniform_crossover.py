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


class UniformCrossover(OP):
    def __init__(self, population_fraction, crossover_probability, **kwargs):
        """Perform uniform crossovers on a given population.

        Args:
            population_fraction (float): How many chromosomes
            should have a cross-over.

            crossover_probability (list(float)): What fraction of the
            gene should be affected by crossovers.

            debug (bool, optional): print debug information and function
            returns additional data.

        Returns:
            tensor: population with a crossover.

        See:
            https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)
        """

        if not (0 < population_fraction <= 1.0):
            raise ValueError("population_fraction must be in ]0. 1]")

        for val in crossover_probability:
            if not (0 < val <= 1.0):
                raise ValueError(
                    "values in crossover_probability must be between ]0. 1]")

        self.population_fraction = population_fraction
        self.crossover_probability = crossover_probability
        self.x_matrix = None
        self.has_cache = False
        self.mutation_shape = []
        super(UniformCrossover, self).__init__(**kwargs)

    def call(self, populations):
        results = []
        for population in populations:
            results.append(self.compute(population))
        return results

    def compute(self, population):

        # mix genomes
        population_copy = B.copy(population)
        population_copy = B.shuffle(population_copy)

        # how many chromosomes to crossover?
        num_crossovers = int(population.shape[0] * self.population_fraction)
        self.print_debug("population size %s" % population.shape[0])
        self.print_debug("num_crossovers %s" % num_crossovers)

        # crossover matrix
        if not self.has_cache:
            self.print_debug("Creating x_matrix and mutation_matrix")
            self.has_cache = True
            self.x_matrix = B.zeros(population.shape, dtype=B.intx())

            # We need to accounting for the fact that the population
            # can be of rank N which makes the fancy indexing painful.
            # we need a shape for the mutation which is this:
            # [num_crossover, num_mutation, ..., num_mutations]
            mutations_shape = [num_crossovers]
            for idx, frac in enumerate(self.crossover_probability):
                num_genes = int(self.x_matrix.shape[idx + 1] * frac)
                mutations_shape.append(num_genes)
            self.print_debug("mutation_shape: %s" % mutations_shape)
            self.mutation_shape = mutations_shape

            # create tensor
            mutations = B.ones(mutations_shape)

            # compute the fancy indexing dynamically
            slices = []
            for size in mutations_shape:
                slices.append(slice(0, size))
            slices = tuple(slices)

            # injecting mutations
            self.x_matrix = B.assign(self.x_matrix, mutations, slices)
        else:
            self.print_debug("Using cached matrix")

        self.x_matrix = B.full_shuffle(self.x_matrix)

        # invert crossover matrix
        inv_x_matrix = B.abs((self.x_matrix) - 1)

        # copy chromosomes that stays the same
        population = population * inv_x_matrix

        # add the mutations
        population += (population_copy * self.x_matrix)

        return population


class UniformCrossover1D(UniformCrossover):
    def __init__(self,
                 population_fraction=0.9,
                 crossover_probability=0.2,
                 **kwargs):
        if not isinstance(crossover_probability, float):
            raise ValueError('crossover_probability must be a float')

        super(UniformCrossover1D,
              self).__init__(population_fraction=population_fraction,
                             crossover_probability=[crossover_probability],
                             **kwargs)


class UniformCrossover2D(UniformCrossover):
    def __init__(self,
                 population_fraction=0.9,
                 crossover_probability=(0.2, 0.2),
                 **kwargs):

        if len(crossover_probability) != 2:
            raise ValueError('crossover_size_fraction must be of form (x, y)')
        super(UniformCrossover2D,
              self).__init__(population_fraction=population_fraction,
                             crossover_probability=crossover_probability,
                             **kwargs)


class UniformCrossover3D(UniformCrossover):
    def __init__(self,
                 population_fraction=0.9,
                 crossover_probability=(0.2, 0.2, 0.2),
                 **kwargs):
        if len(crossover_probability) != 3:
            raise ValueError(
                'crossover_probability must be of form (x, y, z)')  # noqa
        super(UniformCrossover3D,
              self).__init__(population_fraction=population_fraction,
                             crossover_probability=crossover_probability,
                             **kwargs)


if __name__ == '__main__':
    from copy import copy
    GENOME_SHAPE = (10, 4, 4)
    population = B.randint(0, 1024, GENOME_SHAPE)
    population_fraction = 0.5
    crossover_probability = (0.5, 0.5)

    print(population.shape)
    # peforming crossover
    original_population = copy(population)
    population = UniformCrossover2D(population_fraction,
                                    crossover_probability)(population)

    # diff matrix
    diff = B.clip(abs(population - original_population), 0, 1)
    print(diff)
    # expected mutated chromosomes
    expected_mutated = population.shape[0] * population_fraction
    cprint(
        "Expected mutated chromosome:%d (+/- 1 due to collision)" %
        (expected_mutated), 'cyan')
    # select mutated chromosomes
    mutated_chromosomes = []
    for c in diff:
        if B.max(c):
            mutated_chromosomes.append(c)
    cprint("mutated chromosome:%d" % (len(mutated_chromosomes)), 'magenta')

    cprint("[example of mutated chromosome]", 'yellow')
    cprint(mutated_chromosomes[0], 'cyan')

    OP = UniformCrossover2D(population_fraction, crossover_probability)
    res = OP(population)
    for _ in range(100):
        prev = copy(res)
        res = OP(population)
        # acumulating diff matrix
        diff += B.clip(abs(res - prev), 0, 1)

    cprint("[cumulative diff matrix 100 runs]", 'yellow')
    cprint(diff[0], 'cyan')
