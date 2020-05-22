import argparse
from time import time
from termcolor import cprint
from tqdm.auto import tqdm
from benchmark.logger import Logger
from benchmark.setup import setup


def bench(sys_info, logger):
    # ! keep benchmark functions import here so backend selection works
    from benchmark.maxone import solve_onmax_1d, solve_onmax_2d
    from benchmark.tsp import solve_tsp, tsp_setup
    from benchmark.build_op_tests import build_op_test

    GENS = 10
    NUM_RUNS = 3
    TESTS = []  # group, name, function, fn_args, input_shape
    SHAPE1D = [(100, 100), (100, 1000), (1000, 1000)]
    SHAPE2D = [(100, 10, 10), (100, 100, 10), (100, 100, 100)]
    SHAPE3D = [(100, 10, 10, 10), (100, 100, 10, 10), (100, 100, 100, 10)]

    # Ops
    for shape in SHAPE1D:
        TESTS.extend(build_op_test(shape))

    for shape in SHAPE2D:
        TESTS.extend(build_op_test(shape))

    for shape in SHAPE3D:
        TESTS.extend(build_op_test(shape))

    # TSP setup
    for num_cities in [10, 50, 100, 200]:
        shape = (num_cities * 5, num_cities)
        args = [shape, GENS]
        args.extend(tsp_setup(num_cities))
        TESTS.append(['Problem', 'TSP', '1D', solve_tsp, args, shape])

    # onemax 1D
    for shape in SHAPE1D:
        TESTS.append(
            ['Problem', 'OneMax', '1D', solve_onmax_1d, [shape, GENS], shape])

    # onemax 2d
    for shape in SHAPE2D:
        TESTS.append(
            ['Problem', 'OneMax', '2D', solve_onmax_2d, [shape, GENS], shape])

    for test in tqdm(TESTS):
        test_type, group, name, fn, args, shape = test
        # cprint('[%s]%s:%s]' % (group, name, str(shape)), 'yellow')

        timings = []
        for _ in range(NUM_RUNS):
            # tf.keras.backend.clear_session()
            start_time = time()
            fn(args)
            timings.append(time() - start_time)

        logger.record_test(test_type, group, name, timings, NUM_RUNS, GENS,
                           shape)


if __name__ == '__main__':
    BACKENDS = ['tensorflow-cpu', 'tensorflow-gpu', 'cupy', 'numpy']
    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument(
        '--backend',
        '-b',
        help='Backend to use: tensorflow-cpu, tensorflow-gpu, cupy, numpy')
    args = parser.parse_args()

    if not args.backend or args.backend not in BACKENDS:
        parser.print_usage()
        quit()

    sys_info = setup(args.backend)
    logger = Logger(sys_info, args.backend)
    bench(sys_info, logger)
    logger.summary()
