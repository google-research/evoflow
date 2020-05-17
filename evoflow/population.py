import evoflow.backend as B


def randint_population(shape, max_value, min_value=0):
    """Generate a random  population made of Integers

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


def uniform_population(shape, dtype=B.intx()):
    """Generate a uniform population made of Integers. Uniform means that
    each chromosome contains only one time each value and each chromosome
    have them in different order.

    Args:
        (set of ints): shape of the population. Its of the form
        (num_chromosomes, chromosome size)

        dtype (str): tensor type

    Returns:
        Tensor: uniform population.
    """
    population = []
    chromosome = B.range(shape[1], dtype=dtype)
    for i in range(shape[0]):
        chromosome = B.shuffle(chromosome)
        population.append(chromosome)
    return B.tensor(population)
