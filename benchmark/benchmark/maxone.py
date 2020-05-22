from evoflow.engine import EvoFlow
from evoflow.ops import Input, RandomMutations1D, UniformCrossover1D
from evoflow.ops import RandomMutations2D, UniformCrossover2D
from evoflow.selection import SelectFittest
from evoflow.fitness import Sum
from evoflow.population import randint_population


def solve_onmax_1d(args):
    """Solve the onemax problem
    """

    population_shape, generations = args
    population = randint_population(population_shape, 1)
    inputs = Input(population_shape)
    x = RandomMutations1D(max_gene_value=1)(inputs)
    outputs = UniformCrossover1D()(x)
    gf = EvoFlow(inputs, outputs, debug=0)
    fitness_function = Sum(max_sum_value=10000000)
    evolution_strategy = SelectFittest()
    gf.compile(evolution_strategy, fitness_function)
    gf.evolve(population, generations=generations, verbose=0)


def solve_onmax_2d(args):
    """Solve the onemax problem
    """
    population_shape, generations = args
    population = randint_population(population_shape, 1)
    inputs = Input(population_shape)
    x = RandomMutations2D(max_gene_value=1)(inputs)
    outputs = UniformCrossover2D()(x)
    gf = EvoFlow(inputs, outputs, debug=0)
    fitness_function = Sum(max_sum_value=10000000)
    evolution_strategy = SelectFittest()
    gf.compile(evolution_strategy, fitness_function)
    gf.evolve(population, generations=generations, verbose=0)
