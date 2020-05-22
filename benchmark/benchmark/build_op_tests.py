def build_op_test(shape):
    from evoflow.population import randint_population
    from .ops.random_mutation import bench_random_mutation
    from .ops.uniform_crossover import bench_uniform_crossover
    from .ops.single_crossover import bench_single_crossover
    from .ops.dual_crossover import bench_dual_crossover
    from .ops.reverse import bench_reverse
    from .ops.shuffle import bench_shuffle
    from .utils import shape2opdim

    TESTS = []
    population = randint_population(shape, max_value=255)
    OP_DIM = shape2opdim(shape)

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
