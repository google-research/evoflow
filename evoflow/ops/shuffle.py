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


class Shuffle(OP):
    def __init__(self, population_fraction, **kwargs):
        """Shuffle genes within the chromsome.

        Args:
            population_fraction (float): How many chromosomes
            should have a cross-over.

            debug (bool, optional): print debug information and function
            returns additional data.

        Returns:
            tensor: population with shuffled genes.

        See:
            https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)
        """

        if not (0 < population_fraction <= 1.0):
            raise ValueError("num_crossover_fraction must be in [0, 1]")

        self.population_fraction = population_fraction
        super(Shuffle, self).__init__(**kwargs)

    def call(self, populations):
        results = []
        for population in populations:
            results.append(self.compute(population))
        return results

    def compute(self, population):

        # shuffle to make sure  don't hit the same everytime
        if not self.debug:
            population = B.shuffle(population)

        # how many chromosomes to shuffle
        num_shuffle = int(population.shape[0] * self.population_fraction)

        shuffled_population = population[:num_shuffle]
        shuffled_population = B.full_shuffle(shuffled_population)
        self.print_debug("shuffled population", shuffled_population.shape)

        # recompose with non shuffled population
        shuffled_population = B.concatenate(
            [shuffled_population, population[num_shuffle:]])

        return shuffled_population


if __name__ == '__main__':
    from copy import copy
    GENOME_SHAPE = (6, 4, 4)
    population = B.randint(0, 256, GENOME_SHAPE)
    population_fraction = 0.5
    max_crossover_size_fraction = (0.5, 0.5)
    print(population.shape)
    original_population = copy(population)
    population = Shuffle(population_fraction, debug=True)(population)

    # diff matrix
    diff = B.clip(abs(population - original_population), 0, 1)
    print(diff)
