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
from termcolor import cprint


class RandomMutations(OP):
    def __init__(self,
                 population_fraction,
                 mutations_probability,
                 min_gene_value=0,
                 max_gene_value=None,
                 min_mutation_value=1,
                 max_mutation_value=1,
                 use_tf=False,
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
        self.use_tf = use_tf
        super(RandomMutations, self).__init__(**kwargs)

        # specifiy which optimization are available and useful.
        self.TF_FN = True  # support tf.function and needs it
        self.TF_XLA = True  # support XLA compile and needs it

    def call(self, population):

        # three tensors:
        # m1 mutations + padding (affected population)
        # m2 = m1 + zero() (population size)
        affected_population = int(population.shape[0] *
                                  self.population_fraction)
        self.print_debug('affected_population', affected_population)
        # compute number of mutation
        mask_shape = [affected_population]
        num_mutations = affected_population
        total_size = affected_population
        for idx, dim_size in enumerate(population.shape[1:]):
            midx = idx - 1  # recall dimension 0 is the population
            mask_shape.append(dim_size)
            total_size *= dim_size

            # how many mutation
            dim_mutations = int(dim_size * self.mutations_probability[midx])
            num_mutations *= dim_mutations

        # draw mutations
        mutations = B.randint(self.min_mutation_value,
                              self.max_mutation_value + 1,
                              shape=(num_mutations))
        self.print_debug('mutations shape', mutations.shape)

        # padding of zero for the affect population
        padding_size = total_size - num_mutations
        non_mutations = B.zeros((padding_size), dtype=population.dtype)
        self.print_debug('non mutations', non_mutations.shape)

        # construct the mask
        self.print_debug('mask shape', mask_shape)
        mask = B.concatenate([mutations, non_mutations])
        mask = B.shuffle(mask)
        mask = B.reshape(mask, shape=mask_shape)
        self.print_debug('mutation mask', mask)

        # extra padding to match population size if needed
        non_affected_population = population.shape[0] - affected_population
        if non_affected_population:
            padding_size = [non_affected_population]
            padding_size.extend([x for x in population.shape[1:]])
            padding = B.zeros((padding_size), dtype=population.dtype)
            mask = B.concatenate([mask, padding])

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
    from perfcounters import PerfCounters
    pop_shape = (100, 100, 100)
    max_gene_value = 10
    min_gene_value = 0
    population_fraction = 1
    mutations_probability = (0.5, 0.5)
    min_mutation_value = 1
    max_mutation_value = 1

    population = B.randint(0, max_gene_value, pop_shape)

    RM = RandomMutations2D(population_fraction=population_fraction,
                           mutations_probability=mutations_probability,
                           min_gene_value=min_gene_value,
                           max_gene_value=max_gene_value,
                           min_mutation_value=min_mutation_value,
                           max_mutation_value=max_mutation_value,
                           optimization_level=0)

    TF_RM = RandomMutations2D(population_fraction=population_fraction,
                              mutations_probability=mutations_probability,
                              min_gene_value=min_gene_value,
                              max_gene_value=max_gene_value,
                              min_mutation_value=min_mutation_value,
                              max_mutation_value=max_mutation_value,
                              optimization_level=1)

    XLA_RM = RandomMutations2D(population_fraction=population_fraction,
                               mutations_probability=mutations_probability,
                               min_gene_value=min_gene_value,
                               max_gene_value=max_gene_value,
                               min_mutation_value=min_mutation_value,
                               max_mutation_value=max_mutation_value,
                               optimization_level=2)

    TF_RM(population)
    XLA_RM(population)

    cprint('[RandomMutation micro benchmark]', 'yellow')

    ops = [RM, TF_RM, XLA_RM]
    cnts = PerfCounters()
    for idx, op in enumerate(ops):
        cname = 'Optimization level: %d' % idx
        cnts.start(cname)
        for _ in range(3):
            op(population)
        cnts.stop(cname)
    cnts.report()
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
