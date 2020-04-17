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

"Mutations functions"
from geneflow.engine import OP
from geneflow import backend as B
from geneflow.utils import unbox
from termcolor import cprint


class RandomMutations(OP):

    def __init__(self, max_gene_value=None, min_mutation_value=0,
                 max_mutation_value=1, **kwargs):
        """Perform random mutations

        Args:
            max_gene_value = (int, optional): max value a gene can take.
            min_mutation_value (int, optional): Minimal value the mutation
            can take, can be negative. Defaults to 0.
            max_mutation_value (int, optional): Max value the mutations can
            take. Defaults to 1.
            debug (bool, optional): Defaults to False.
        """
        self.max_gene_value = max_gene_value
        self.min_mutation_value = min_mutation_value
        self.max_mutation_value = max_mutation_value
        self.mask_exist = False
        self.mask = None  # init at the first call
        super(RandomMutations, self).__init__(**kwargs)

    def call(self, chromosomes_population):
        results = []

        for chromosomes in chromosomes_population:
            if self.mask_exist:
                self.print_debug('reusing mask %s' % str(self.mask.shape))
                B.shuffle(self.mask)
            else:
                # NOTE: we assume we only support a single mask shape per op.
                self.print_debug('creating mask %s' % str(chromosomes.shape))
                self.mask = self._generate_random_mask(chromosomes.shape)
                self.mask_exist = True

            chromosomes = chromosomes + self.mask

            if self.max_gene_value:
                chromosomes %= self.max_gene_value + 1
            results.append(chromosomes)
        return unbox(results)

    def _generate_random_mask(self, shape):
        mask = B.zeros(shape)
        for row_id in range(shape[0]):
            val = 0

            # ensure we have a valid mutation value
            while not val:
                val = B.randint(self.min_mutation_value,
                                self.max_mutation_value + 1)
            mask[row_id][B.randint(shape[1])] = val

        return B.tensor(mask)


if __name__ == '__main__':
    from copy import copy
    W = 8
    NUM_GENES = 100
    GENOME_SHAPE = (W, W)

    min_mutation_value = -3
    max_mutation_value = 3

    chromosomes = B.randint(0, NUM_GENES, GENOME_SHAPE)
    RM = RandomMutations(max_gene_value=100,
                         min_mutation_value=min_mutation_value,
                         max_mutation_value=max_mutation_value,
                         debug=True)
    RM(chromosomes)

    chromosomes_sav = copy(chromosomes)
    cprint('[Initial genepool]', 'blue')
    cprint(chromosomes_sav, 'blue')
    chromosomes = RM(chromosomes)

    cprint('\n[Mutated genepool]', 'yellow')
    cprint(chromosomes, 'yellow')

    cprint('\n[Diff]', 'magenta')
    diff = chromosomes - chromosomes_sav
    cprint(diff, 'magenta')

    assert B.max(diff) <= max_mutation_value
    assert B.min(diff) >= min_mutation_value
