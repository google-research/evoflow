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
from geneflow.engine import OP
from geneflow import backend as B
from tabulate import tabulate


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

        if not (0 < num_crossover_fraction <= 1.0):
            raise ValueError("num_crossover_fraction must be in ]0. 1]")

        for val in crossover_size_fraction:
            if not (0 < num_crossover_fraction <= 1.0):
                raise ValueError(
                    "values in crossover_size_fraction must be between ]0. 1]")

        self.num_crossover_fraction = num_crossover_fraction
        self.crossover_size_fraction = crossover_size_fraction
        self.x_matrix = None
        self.has_cache = False
        self.mutation_shape = []
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

        # how many chromosomes to crossover?
        num_crossover_chromosomes = int(chromosomes.shape[0] *
                                        self.num_crossover_fraction)
        self.print_debug("num_crossover_chromosome %s" %
                         num_crossover_chromosomes)

        # crossover matrix
        if not self.has_cache:
            self.print_debug("Creating x_matrix and mutation_matrix")
            self.has_cache = True
            self.x_matrix = B.zeros(chromosomes.shape, dtype='int')

            # We need to accounting for the fact that the population
            # can be of rank N which makes the fancy indexing painful.
            # we need a shape for the mutation which is this:
            # [num_crossover, num_mutation, ..., num_mutations]
            mutations_shape = [num_crossover_chromosomes]
            for idx, frac in enumerate(self.crossover_size_fraction):
                num_genes = int(self.x_matrix.shape[idx + 1] * frac)
                mutations_shape.append(num_genes)
            self.print_debug("mutation_shape: %s" % mutations_shape)
            self.mutation_shape = mutations_shape

            # mutations_shape = B.tensor(mutations_shape, dtype='int')
            mutations = B.ones(mutations_shape)

            # compute the fancy indexing dynamically
            slices = [slice(0, num_crossover_chromosomes, 1)]
            for size in mutations_shape[1:]:
                slices.append(slice(0, size, 1))
            slices = tuple(slices)

            # injecting mutations
            self.x_matrix[slices] = mutations

            # shuffling preserve structure. Here we need to shuffle
            # all chromsomes dimensions so we will flatten the matrix
            # and shuffle its transpose before recreating it matrix
            total_chromosome_size = B.prod(B.tensor(self.x_matrix.shape[1:]))
            flatten_shape = (self.x_matrix.shape[0],
                             int(total_chromosome_size))
            self.flatten_x_matrix = B.reshape(self.x_matrix, flatten_shape)
        else:
            self.print_debug("Using cache")

        B.shuffle(self.flatten_x_matrix)
        B.shuffle(self.flatten_x_matrix.T)

        self.x_matrix = B.reshape(self.flatten_x_matrix, self.x_matrix.shape)

        # invert crossover matrix
        inv_x_matrix = B.abs((self.x_matrix) - 1)

        # copy chromosomes that stays the same
        mutated_chromosomes = chromosomes * inv_x_matrix

        # add the mutations
        mutated_chromosomes += (chromosomes2 * self.x_matrix)

        if self.debug:
            return {'mutated': mutated_chromosomes,
                    'original': ref_chromosomes,
                    'num_crossover_chromosome': num_crossover_chromosomes,

                    }

        return mutated_chromosomes


class UniformCrossover1D(UniformCrossover):

    def __init__(self, num_crossover_fraction=0.9,
                 crossover_size_fraction=0.2, **kwargs):

        super(UniformCrossover1D, self).__init__(
            num_crossover_fraction=num_crossover_fraction,
            crossover_size_fraction=[crossover_size_fraction],
            **kwargs)


class UniformCrossover2D(UniformCrossover):

    def __init__(self, num_crossover_fraction=0.9,
                 crossover_size_fraction=(0.2, 0.2), **kwargs):

        super(UniformCrossover2D, self).__init__(
            num_crossover_fraction=num_crossover_fraction,
            crossover_size_fraction=crossover_size_fraction,
            **kwargs)


class UniformCrossover3D(UniformCrossover):

    def __init__(self, num_crossover_fraction=0.9,
                 crossover_size_fraction=(0.2, 0.2, 0.2), **kwargs):

        super(UniformCrossover3D, self).__init__(
            num_crossover_fraction=num_crossover_fraction,
            crossover_size_fraction=crossover_size_fraction,
            **kwargs)


if __name__ == '__main__':
    print(B.backend())
    GENOME_SHAPE = (10, 4, 4)
    chromosomes = B.randint(0, 1024, GENOME_SHAPE)
    num_crossover_fraction = 0.5
    max_crossover_size_fraction = (0.5, 0.5)

    print(chromosomes.shape)
    # peforming crossover
    res = UniformCrossover2D(num_crossover_fraction,
                             max_crossover_size_fraction,
                             debug=True)(chromosomes)

    # diff matrix
    diff = B.clip(abs(res['mutated'] - res['original']), 0, 1)

    # expected mutated chromosomes
    expected_mutated = chromosomes.shape[0] * num_crossover_fraction
    cprint("Expected mutated chromosome:%d (+/- 1 due to collision)" %
           (expected_mutated), 'cyan')
    # select mutated chromosomes
    mutated_chromosomes = []
    for c in diff:
        if B.max(c):
            mutated_chromosomes.append(c)
    cprint(mutated_chromosomes[0], 'cyan')

    cprint("mutated chromosome:%d" % (len(mutated_chromosomes)), 'magenta')

    bins = B.bincount(diff.flatten())
    mutation_rate = int(bins[1] / sum(bins) * 100)

    # expected mutated rows
    mutated_rows = max_crossover_size_fraction[0] * GENOME_SHAPE[1]
    cprint('expected mutated row:%d' % mutated_rows, 'cyan')
    mutated_cells = max_crossover_size_fraction[0] * GENOME_SHAPE[2]
    cprint('expected mutated cell:%d' % mutated_cells, 'cyan')

    # actual mutated rows:
    rows = []
    for cidx, c in enumerate(mutated_chromosomes):
        mr = 0
        for r in c:
            mc = B.sum(r)
            if mc:
                mr += 1
        rows.append([cidx, mr, mc])
    print(tabulate(rows, headers=["Chromosome", "Row mutated",
                                  "Cell per row mutated"]))

    for _ in range(100):
        res = UniformCrossover2D(num_crossover_fraction,
                                 max_crossover_size_fraction,
                                 debug=1)(chromosomes)
        # acumulating diff matrix
        diff += B.clip(abs(res['mutated'] - res['original']), 0, 1)

    cprint("[cumulative diff matrix]", 'yellow')
    cprint(diff[0], 'cyan')
