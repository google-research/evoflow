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

from evoflow.engine import OP
from evoflow import backend as B


class DualCrossover(OP):
    def __init__(self, population_fraction, crossover_probability, **kwargs):
        """Perform Dual crossovers on a given population.

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
                    "crossover_size_fraction values must be between ]0. 1]")

        self.population_fraction = population_fraction
        self.crossover_probability = crossover_probability
        self.has_cache = False
        self.mutation_shape = []
        self.slices = None
        super(DualCrossover, self).__init__(**kwargs)

    def call(self, populations):
        results = []
        for population in populations:
            results.append(self.compute(population))
        return results

    def compute(self, population):

        # mix genomes
        population_copy = B.copy(population)
        population_copy = B.shuffle(population_copy)

        # how many chromosomes to crossover
        num_crossover_chromosomes = int(population.shape[0] *
                                        self.population_fraction)

        self.print_debug('num chromosomes', num_crossover_chromosomes)

        if not self.has_cache:
            self.print_debug("Caching fancy indexing")
            self.has_cache = True

            # compute the shape needed for the mutation
            mutations_shape = [num_crossover_chromosomes]
            for idx, frac in enumerate(self.crossover_probability):
                num_genes = int(population.shape[idx + 1] * frac)
                mutations_shape.append(num_genes)
            self.mutations_shape = mutations_shape
            self.print_debug("population_shape:", population.shape)
            self.print_debug("mutation_shape:", self.mutations_shape)

        else:
            self.print_debug("Using cached fancy indexing")

        # compute the fancy indexing dynamlically
        # ! the start point must be randomized
        slices = [slice(0, num_crossover_chromosomes)]
        for idx, crossover_size in enumerate(self.mutations_shape[1:]):
            # ! making indexing explicit as its a huge pitfall
            mutation_dim = idx + 1
            max_start = population.shape[mutation_dim] - crossover_size + 1
            start = B.randint(0, max_start, 1)[0]
            slices.append(slice(start, crossover_size + start))

        slices = tuple(slices)
        self.print_debug(slices)

        # crossover
        cross_section = population_copy[slices]
        population = B.assign(population, cross_section, slices)

        return population


class DualCrossover1D(DualCrossover):
    def __init__(self,
                 population_fraction=0.9,
                 crossover_probability=0.2,
                 **kwargs):
        if not isinstance(crossover_probability, float):
            raise ValueError('crossover_probability must be a float')

        super(DualCrossover1D,
              self).__init__(population_fraction=population_fraction,
                             crossover_probability=[crossover_probability],
                             **kwargs)


class DualCrossover2D(DualCrossover):
    def __init__(self,
                 population_fraction=0.9,
                 crossover_probability=(0.2, 0.2),
                 **kwargs):

        if len(crossover_probability) != 2:
            raise ValueError('crossover_probability must be of form (x, y)')
        super(DualCrossover2D,
              self).__init__(population_fraction=population_fraction,
                             crossover_probability=crossover_probability,
                             **kwargs)


class DualCrossover3D(DualCrossover):
    def __init__(self,
                 population_fraction=0.9,
                 crossover_probability=(0.2, 0.2, 0.2),
                 **kwargs):

        if len(crossover_probability) != 3:
            raise ValueError('crossover_probability must be of form (x, y, z)')

        super(DualCrossover3D,
              self).__init__(population_fraction=population_fraction,
                             crossover_probability=crossover_probability,
                             **kwargs)


if __name__ == '__main__':
    from copy import copy
    GENOME_SHAPE = (6, 4, 4)
    population = B.randint(0, 256, GENOME_SHAPE)
    population_fraction = 0.5
    max_crossover_size_fraction = (0.5, 0.5)
    print(population.shape)
    original_population = copy(population)
    population = DualCrossover2D(population_fraction,
                                 max_crossover_size_fraction,
                                 debug=True)(population)

    # diff matrix
    diff = B.clip(abs(population - original_population), 0, 1)
    print(diff)
