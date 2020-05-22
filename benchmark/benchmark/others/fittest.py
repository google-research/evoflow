from evoflow.fitness import Sum
from evoflow.selection import SelectFittest
from evoflow.population import randint_population
from ..utils import shape2opdim


def build_fittest_test(shape):
    population = randint_population(shape, 256)
    # test_type, group, name, fn, args, shape = test
    # single test for now but still return a list(test)
    OP_DIM = shape2opdim(shape)
    return [['Selection', 'Fittest', OP_DIM, bench_fittest, population, shape]]


def bench_fittest(population):
    fitness_function = Sum()
    selector = SelectFittest()
    selected_pop, fitness_scores, metrics = selector(fitness_function,
                                                     population, population)
