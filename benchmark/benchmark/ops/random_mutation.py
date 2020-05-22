from evoflow.ops import RandomMutations1D, RandomMutations2D
from evoflow.ops import RandomMutations3D


def bench_random_mutation(population):

    # setup
    shape = population.shape
    max_gene_value = 10
    min_gene_value = 0
    population_fraction = 1
    min_mutation_value = 1
    max_mutation_value = 1

    # select the right op
    if len(shape) == 2:
        mutations_probability = 0.5
        OP = RandomMutations1D
    elif len(shape) == 3:
        mutations_probability = (0.5, 0.5)
        OP = RandomMutations2D
    elif len(shape) == 4:
        mutations_probability = (0.5, 0.5, 0.5)
        OP = RandomMutations3D
    else:
        raise ValueError("too many dimensions")

    OP(population_fraction=population_fraction,
       mutations_probability=mutations_probability,
       min_gene_value=min_gene_value,
       max_gene_value=max_gene_value,
       min_mutation_value=min_mutation_value,
       max_mutation_value=max_mutation_value)(population)
