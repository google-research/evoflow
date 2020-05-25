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

from evoflow.utils import slices2array
from evoflow.engine import OP
from evoflow import backend as B


class SingleCrossover(OP):
    O_AUTOGRAPH = False
    O_XLA = False

    def __init__(self, population_fraction, max_crossover_probability,
                 **kwargs):
        """Perform single crossovers on a given population.

        Args:
            population_fraction (float): How many chromosomes
            should have a cross-over.

            max_crossover_probability (list(float)): What is the maximum
            fraction of the genes that will be affected by the crossover.

            debug (bool, optional): print debug information and function
            returns additional data.

        Returns:
            tensor: population with a crossover

        See:
            https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)
        """

        if not (0 < population_fraction <= 1.0):
            raise ValueError("num_crossover_fraction must be in ]0. 1]")

        for val in max_crossover_probability:
            if not (0 < val <= 1.0):
                raise ValueError(
                    "max_crossover_probability values must be between ]0. 1]")

        self.population_fraction = population_fraction
        self.max_crossover_probability = max_crossover_probability
        super(SingleCrossover, self).__init__(**kwargs)

    def call(self, population):

        # mix genomes
        shuffled_population = B.copy(population)
        shuffled_population = B.shuffle(shuffled_population)

        # how many chromosomes to crossover
        num_crossovers = int(population.shape[0] * self.population_fraction)
        self.print_debug('num_crossovers', num_crossovers)

        # compute the shape needed for the mutation
        mutations_shape = [num_crossovers]
        for idx, frac in enumerate(self.max_crossover_probability):
            max_genes = int(population.shape[idx + 1] * frac + 1)
            if max_genes > 1:
                num_genes = B.randint(1, high=max_genes)
            else:
                num_genes = 1
            mutations_shape.append(num_genes)
        self.print_debug("mutation_shape: %s" % mutations_shape)

        slices = []
        for crossover_size in mutations_shape:
            slices.append(slice(0, crossover_size))
        slices = tuple(slices)
        tslices = slices2array(slices)
        self.print_debug('slices', slices)

        # crossover
        cross_section = shuffled_population[slices]
        population = B.assign(population, cross_section, tslices)

        return population


class SingleCrossover1D(SingleCrossover):
    def __init__(self,
                 population_fraction=0.9,
                 max_crossover_probability=0.2,
                 **kwargs):
        if not isinstance(max_crossover_probability, float):
            raise ValueError('max_crossover_probability must be a float')

        super(SingleCrossover1D, self).__init__(
            population_fraction=population_fraction,
            max_crossover_probability=[max_crossover_probability],
            **kwargs)


class SingleCrossover2D(SingleCrossover):
    def __init__(self,
                 population_fraction=0.9,
                 max_crossover_probability=(0.2, 0.2),
                 **kwargs):

        if len(max_crossover_probability) != 2:
            raise ValueError(
                'max_crossover_probability must be of form (x, y)')

        super(SingleCrossover2D, self).__init__(
            population_fraction=population_fraction,
            max_crossover_probability=max_crossover_probability,
            **kwargs)


class SingleCrossover3D(SingleCrossover):
    def __init__(self,
                 population_fraction=0.9,
                 max_crossover_probability=(0.2, 0.2, 0.2),
                 **kwargs):

        if len(max_crossover_probability) != 3:
            raise ValueError(
                'max_crossover_probability must be of form (x, y, z)')

        super(SingleCrossover3D, self).__init__(
            population_fraction=population_fraction,
            max_crossover_probability=max_crossover_probability,
            **kwargs)


if __name__ == '__main__':
    from copy import copy
    from evoflow.utils import op_optimization_benchmark

    NUM_RUNS = 3
    pop_shape = (1000, 100, 100)
    population = B.randint(0, 256, pop_shape)
    population_fraction = 0.5
    max_reverse_probability = (0.5, 0.5)

    OP = SingleCrossover2D(population_fraction, max_reverse_probability)
    op_optimization_benchmark(population, OP, NUM_RUNS).report()
    quit()

    GENOME_SHAPE = (6, 4, 4)
    population = B.randint(0, 256, GENOME_SHAPE)
    population_fraction = 0.5
    max_crossover_size_fraction = (0.5, 0.5)
    print(population.shape)
    original_population = copy(population)
    population = SingleCrossover2D(population_fraction,
                                   max_crossover_size_fraction,
                                   debug=True)(population)

    # diff matrix
    diff = B.clip(abs(population - original_population), 0, 1)
    print(diff)
