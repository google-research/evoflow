import evoflow.backend as B


def genRandIntPopulation(shape, max_value, min_value=0):
    """Generate a random population of Chromosome made of Integers

    Args:
        (set of ints): shape of the population. Its of the form
        (num_chromosomes, chromosome_dim_1, .... chromesome_dim_n)

        max_value (int): Maximum value taken by a given gene.

        min_value (int, optional): Min value a gene can take. Defaults to 0.

    Returns:
        Tensor: random population.
    """
    high = max_value + 1
    return B.randint(low=min_value, high=high, shape=shape)
