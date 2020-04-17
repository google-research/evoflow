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

"crossover functions"
from termcolor import cprint
from geneflow.engine import OP
from geneflow import backend as B


class UniformCrossover(OP):

    def __init__(self, num_crossover_fraction=0.9,
                 crossover_size_fraction=0.2, **kwargs):
        """Perform uniform crossovers on a given population.

        Args:
            num_crossover_fraction (float, optional): How many chromosomes
            should have a cross-over. Defaults to 0.9 (90%).
            crossover_size_fraction (float, optional): What fraction of the
            gene should be affected by the crossover. Defaults to 0.2 (20%).
            debug (bool, optional): [description]. Defaults to False.
        """

        self.num_crossover_fraction = num_crossover_fraction
        self.crossover_size_fraction = crossover_size_fraction
        super(UniformCrossover, self).__init__(**kwargs)

    def call(self, chromosome_populations):
        results = []
        for chromosomes in chromosome_populations:
            results.append(self.compute(chromosomes))
        return results

    def compute(self, chromosomes):

        # mix genomes
        B.shuffle(chromosomes)
        chromosomes2 = B.copy(chromosomes)
        B.shuffle(chromosomes2)

        # replace genome
        if self.debug:
            ref_chromosomes = B.copy(chromosomes)

        # how many chromosomes to crossover
        num_chromosomes = int(chromosomes.shape[0] *
                              self.num_crossover_fraction)

        # crossover matrix
        x_matrix = B.zeros((num_chromosomes, chromosomes.shape[1]))

        num_mutations = int(chromosomes.shape[1] *
                            self.crossover_size_fraction)

        x_matrix[:, 0:num_mutations] = B.ones((num_chromosomes, num_mutations))
        B.shuffle(x_matrix.T)

        # invert cross matrix
        inv_x_matrix = B.abs((x_matrix) - 1)

        # copy chromosomes that stays the same
        mutated_chromosomes = chromosomes[:num_chromosomes] * inv_x_matrix

        # add the mutations
        mutated_chromosomes += chromosomes2[:num_chromosomes] * x_matrix
        chromosomes[:num_chromosomes] = mutated_chromosomes

        if self.debug:
            return chromosomes, ref_chromosomes

        return chromosomes


class SingleCrossover(OP):

    def __init__(self, num_crossover_fraction=0.9,
                 crossover_size_fraction=0.2, **kwargs):
        """Perform single crossovers on a given population.

        Args:
            num_crossover_fraction (float, optional): How many chromosomes
            should have a cross-over. Defaults to 0.9 (90%).
            crossover_size_fraction (float, optional): What fraction of the
            genes should be affected by the crossover. Defaults to 0.2 (20%).
            debug (bool, optional): [description]. Defaults to False.
        """

        self.num_crossover_fraction = num_crossover_fraction
        self.crossover_size_fraction = crossover_size_fraction
        super(SingleCrossover, self).__init__(**kwargs)

    def call(self, populations):
        results = []
        for population in populations:
            results.append(self.compute(population))
        return results

    def compute(self, population):

        # mix genomes
        population_copy = B.copy(population)
        B.shuffle(population_copy)

        # how many chromosomes to crossover
        num_chromosomes = int(population.shape[0] *
                              self.num_crossover_fraction)

        # crossover size
        max_size = population.shape[1] * self.crossover_size_fraction + 1
        stop = B.randint(1, max_size)

        if self.debug:
            print('single crossover', 'num chromosomes', num_chromosomes,
                  'crossover size', stop)
            original_population = B.copy(population)

        # crossover
        cross_section = population_copy[:num_chromosomes, 0:stop]
        population[:num_chromosomes, 0:stop] = cross_section

        if self.debug:
            return population, original_population
        return population


class DualCrossover(OP):

    def __init__(self, num_crossover_fraction=0.9,
                 crossover_size_fraction=0.2, **kwargs):
        """Perform single crossovers on a given population.

        Args:
            num_crossover_fraction (float, optional): How many chromosomes
            should have a cross-over. Defaults to 0.9 (90%).
            crossover_size_fraction (float, optional): What fraction of the
            genes should be affected by the crossover. Defaults to 0.2 (20%).
        """

        self.num_crossover_fraction = num_crossover_fraction
        self.crossover_size_fraction = crossover_size_fraction
        super(DualCrossover, self).__init__(**kwargs)

    def call(self, chromosome_populations):
        results = []
        for chromosomes in chromosome_populations:
            results.append(self.compute(chromosomes))
        return results

    def compute(self, chromosomes):

        # mix genomes
        B.shuffle(chromosomes)
        chromosomes2 = B.copy(chromosomes)
        B.shuffle(chromosomes2)

        # how many chromosomes to crossover
        num_chromosomes = int(chromosomes.shape[0] *
                              self.num_crossover_fraction)

        # crossover size
        start = B.randint(1, chromosomes.shape[1] - 2)

        max_size = chromosomes.shape[1] * self.crossover_size_fraction
        max_size += start + 1
        if max_size > chromosomes.shape[1] - 1:
            max_size = start + chromosomes.shape[1] - 1

        stop = B.randint(start + 1, max_size)

        if self.debug:
            print('crossover', 'num chromosomes', num_chromosomes,
                  'crossover size', stop - start, 'start', start,
                  'stop', stop)
            ref_chromosomes = B.copy(chromosomes)

        # crossover
        mutated_section = chromosomes2[:num_chromosomes, start:stop]
        chromosomes[:num_chromosomes, start:stop] = mutated_section

        if self.debug:
            return chromosomes, ref_chromosomes
        return chromosomes


if __name__ == '__main__':
    print(B.backend())
    GENOME_SHAPE = (16, 16)
    chromosomes = B.randint(0, 1024, GENOME_SHAPE)
    num_crossover_fraction = 0.5
    max_crossover_size_fraction = 0.2

    # peforming crossover
    chromosomes, ref_chromosomes = UniformCrossover(num_crossover_fraction,
                                                    max_crossover_size_fraction,  # noqa
                                                    debug=True)(chromosomes)

    # measuring mutation rate
    diff = B.clip(abs(chromosomes - ref_chromosomes), 0, 1)
    cprint(diff, 'cyan')

    bins = B.bincount(diff.flatten())
    mutation_rate = int(bins[1] / sum(bins) * 100)

    # computing expected mutation rate
    expected_mutation_rate = num_crossover_fraction
    expected_mutation_rate *= max_crossover_size_fraction
    expected_mutation_rate *= 100

    # diff
    mutation_rate_diff = abs(mutation_rate - expected_mutation_rate)

    cprint('ratio mutations %.2f%%' % mutation_rate, 'cyan')
    cprint('expected ratio  %.2f%%' % expected_mutation_rate, 'yellow')
    cprint('rate difference %.2f%%' % mutation_rate_diff, 'green')
