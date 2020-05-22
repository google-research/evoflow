def build_op_test(shape):
    from evoflow.population import randint_population
    from benchmark.random_mutation import bench_random_mutation
    from benchmark.uniform_crossover import bench_uniform_crossover
    from benchmark.single_crossover import bench_single_crossover
    from benchmark.dual_crossover import bench_dual_crossover
    from benchmark.reverse import bench_reverse
    from benchmark.shuffle import bench_shuffle

    TESTS = []
    population = randint_population(shape, max_value=255)

    if len(shape) == 2:
        OP_DIM = "1D"
    elif len(shape) == 3:
        OP_DIM = "2D"
    elif len(shape) == 4:
        OP_DIM = "3D"
    else:
        raise ValueError('Too many dimensions')

    TESTS.append(['OP', 'Shuffle', OP_DIM, bench_shuffle, population, shape])

    TESTS.append([
        'OP', 'RandomMutation', OP_DIM, bench_random_mutation, population,
        shape
    ])

    TESTS.append([
        'OP', 'UniformCrossover', OP_DIM, bench_uniform_crossover, population,
        shape
    ])

    TESTS.append([
        'OP', 'SingleCrossover', OP_DIM, bench_single_crossover, population,
        shape
    ])

    TESTS.append([
        'OP', 'DualCrossover', OP_DIM, bench_dual_crossover, population, shape
    ])

    TESTS.append(['OP', 'Reverse', OP_DIM, bench_reverse, population, shape])

    return TESTS
