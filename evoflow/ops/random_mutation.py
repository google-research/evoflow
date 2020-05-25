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
from termcolor import cprint


class RandomMutations(OP):

    O_AUTOGRAPH = False
    O_XLA = False

    def __init__(self,
                 population_fraction,
                 mutations_probability,
                 min_gene_value=0,
                 max_gene_value=None,
                 min_mutation_value=1,
                 max_mutation_value=1,
                 **kwargs):
        """Perform random mutations

        Args:
            population_fraction (float): What fraction of the population should
            be affected by the mutations.

            mutations_probability (list(float)): What fraction of the
            genes should be affected by the mutations.

            min_gene_value = (int, optional): min value a gene can take.
            Defaults to 0.

            max_gene_value = (int, optional): max value a gene can take.
            Defaults to None.

            min_mutation_value (int): Minimal value the mutation
            can take, can be negative. Defaults to 0.

            max_mutation_value (int): Max value the mutations can
            take. Defaults to 1.

            debug (bool, optional): Defaults to False.
        """

        if not (0 < population_fraction <= 1.0):
            raise ValueError("population_fraction must be in ]0. 1]")

        for val in mutations_probability:
            if not (0 < val <= 1.0):
                raise ValueError(
                    "mutations_probability values must be between ]0. 1]")

        if min_gene_value and max_gene_value and min_gene_value >= max_gene_value:  # noqa
            raise ValueError("min_gene_value must < than max_gen_value")

        # ! don't remove the _num_ qualifier. It might feel it is unecessary
        # ! but we want consistent naming accross ops and crossover ops need it
        self.population_fraction = population_fraction
        self.mutations_probability = mutations_probability
        self.min_gene_value = min_gene_value
        self.max_gene_value = max_gene_value
        self.min_mutation_value = min_mutation_value
        self.max_mutation_value = max_mutation_value
        super(RandomMutations, self).__init__(**kwargs)

    def call(self, population):
        """ Create the mask use to generate mutations

        Args:
            population_shape (list): population tensor shape.
        Returns:
            tensor: mask

        Generation works by:
        1. creating a slice that contains the mutation
        2. Inserting it into the mask
        3. Shuffle the mask in every dimension to distribute them
        """
        affected_population = int(population.shape[0] *
                                  self.population_fraction)

        # Build sub tensors & slices by iterating through tensor dimensions
        sub_tensor_shape = [affected_population]
        slices = [slice(0, affected_population)]
        for idx, pop_size in enumerate(population.shape[1:]):
            midx = idx - 1  # recall dim1 are genes.
            max_genes = int(pop_size * self.mutations_probability[midx] + 1)
            num_genes = B.randint(1, high=max_genes)
            sub_tensor_shape.append(num_genes)
            slices.append(slice(0, num_genes))
        slices = tuple(slices)
        tslices = slices2array(slices)
        self.print_debug("sub_tensor_shape", sub_tensor_shape)

        # drawing mutations
        mutations = B.randint(self.min_mutation_value,
                              self.max_mutation_value + 1,
                              shape=sub_tensor_shape)
        # blank mask
        mask = B.zeros(population.shape, dtype=mutations.dtype)

        # add mutations
        # print('mtuation', mutations.dtype)
        # print('mask', mask.dtype)
        mask = B.assign(mask, mutations, tslices)

        # shuffle mask every axis
        mask = B.full_shuffle(mask)

        # mutate
        population = population + mask

        # normalize
        if self.max_gene_value or self.min_gene_value:
            self.print_debug("min_gen_val", self.min_gene_value)
            self.print_debug("max_gen_val", self.max_gene_value)

            population = B.clip(population,
                                min_val=self.min_gene_value,
                                max_val=self.max_gene_value)
        return population


class RandomMutations1D(RandomMutations):
    def __init__(self,
                 population_fraction=0.9,
                 mutations_probability=0.5,
                 min_gene_value=0,
                 max_gene_value=None,
                 min_mutation_value=1,
                 max_mutation_value=1,
                 **kwargs):
        if not isinstance(mutations_probability, float):
            raise ValueError('mutations_probability must be a float')

        super(RandomMutations1D,
              self).__init__(population_fraction,
                             mutations_probability=[mutations_probability],
                             min_gene_value=min_gene_value,
                             max_gene_value=max_gene_value,
                             min_mutation_value=min_mutation_value,
                             max_mutation_value=max_mutation_value,
                             **kwargs)


class RandomMutations2D(RandomMutations):
    def __init__(self,
                 population_fraction=0.9,
                 mutations_probability=(0.5, 0.5),
                 min_gene_value=0,
                 max_gene_value=None,
                 min_mutation_value=1,
                 max_mutation_value=1,
                 **kwargs):
        if len(mutations_probability) != 2:
            raise ValueError('mutations_probability must be of form (x, y)')

        super(RandomMutations2D,
              self).__init__(population_fraction=population_fraction,
                             mutations_probability=mutations_probability,
                             min_gene_value=min_gene_value,
                             max_gene_value=max_gene_value,
                             min_mutation_value=min_mutation_value,
                             max_mutation_value=max_mutation_value,
                             **kwargs)


class RandomMutations3D(RandomMutations):
    def __init__(self,
                 population_fraction=0.9,
                 mutations_probability=(0.5, 0.5, 0.5),
                 min_gene_value=0,
                 max_gene_value=None,
                 min_mutation_value=0,
                 max_mutation_value=1,
                 **kwargs):
        if len(mutations_probability) != 3:
            raise ValueError('mutations_probability must be of form (x, y, z)')

        super(RandomMutations3D,
              self).__init__(population_fraction=population_fraction,
                             mutations_probability=mutations_probability,
                             min_gene_value=min_gene_value,
                             max_gene_value=max_gene_value,
                             min_mutation_value=min_mutation_value,
                             max_mutation_value=max_mutation_value,
                             **kwargs)


if __name__ == '__main__':
    from copy import copy
    from evoflow.utils import op_optimization_benchmark
    NUM_RUNS = 10
    pop_shape = (100, 100, 10)
    max_gene_value = 10
    min_gene_value = 0
    population_fraction = 1
    mutations_probability = (0.5, 0.5)
    min_mutation_value = 1
    max_mutation_value = 1

    population = B.randint(0, max_gene_value, pop_shape)

    OP = RandomMutations2D(
        population_fraction=population_fraction,
        mutations_probability=mutations_probability,
        min_gene_value=min_gene_value,
        max_gene_value=max_gene_value,
        min_mutation_value=min_mutation_value,
        max_mutation_value=max_mutation_value,
    )
    op_optimization_benchmark(population, OP, NUM_RUNS).report()
    quit()
    # display
    pop_shape = (6, 4, 4)
    max_gene_value = 10
    min_gene_value = 0
    population_fraction = 0.5
    mutations_probability = (0.5, 0.5)
    min_mutation_value = 1
    max_mutation_value = 1

    population = B.randint(0, max_gene_value, pop_shape)

    OP = RandomMutations2D(population_fraction=population_fraction,
                           mutations_probability=mutations_probability,
                           min_gene_value=min_gene_value,
                           max_gene_value=max_gene_value,
                           min_mutation_value=min_mutation_value,
                           max_mutation_value=max_mutation_value)

    chromosomes_sav = copy(population)
    cprint('[Initial genepool]', 'blue')
    cprint(chromosomes_sav, 'blue')
    population = OP(population)

    cprint('\n[Mutated genepool]', 'yellow')
    cprint(population, 'yellow')

    cprint('\n[Diff]', 'magenta')
    diff = population - chromosomes_sav
    cprint(diff, 'magenta')

    assert B.max(diff) <= max_mutation_value
