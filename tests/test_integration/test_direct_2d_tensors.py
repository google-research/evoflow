from evoflow.engine import EvoFlow
from evoflow.ops import Input, RandomMutations2D, UniformCrossover2D
from evoflow.selection import SelectFittest
from evoflow.fitness import Sum
from evoflow.population import randint_population
import evoflow.backend as B


def test_direct_2d():
    NUM_EVOLUTIONS = 2
    POPULATION_SIZE = 3
    GENE_SIZE = 4
    MAX_VAL = 10
    SHAPE = (POPULATION_SIZE, GENE_SIZE, GENE_SIZE)
    population = randint_population(SHAPE, MAX_VAL)
    fitness_function = Sum(max_sum_value=GENE_SIZE)
    evolution_strategy = SelectFittest()

    inputs = Input(shape=SHAPE)
    # x = RandomMutations2D(max_gene_value=1, min_gene_value=0)(inputs)
    outputs = UniformCrossover2D()(inputs)

    ef = EvoFlow(inputs, outputs, debug=True)
    ef.compile(evolution_strategy, fitness_function)
    ef.evolve(population, generations=NUM_EVOLUTIONS, verbose=0)


if __name__ == "__main__":
    test_direct_2d()
