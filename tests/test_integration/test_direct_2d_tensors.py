from geneflow.engine import GeneFlow
from geneflow.ops import Input, RandomMutations2D, UniformCrossover2D
from geneflow.selection import SelectFittest
from geneflow.fitness import Sum
import geneflow.backend as B


def test_direct_2d():
    NUM_EVOLUTIONS = 2
    POPULATION_SIZE = 32
    GENE_SIZE = 10
    MAX_VAL = 256
    SHAPE = (POPULATION_SIZE, GENE_SIZE, GENE_SIZE)

    population = B.randint(0, MAX_VAL, SHAPE)
    inputs = Input(shape=SHAPE)
    x = RandomMutations2D(max_gene_value=1, min_gene_value=0)(inputs)
    outputs = UniformCrossover2D()(x)
    gf = GeneFlow(inputs, outputs)

    fitness_function = Sum(max_sum_value=GENE_SIZE)
    evolution_strategy = SelectFittest()

    gf.compile(evolution_strategy, fitness_function)
    gf.evolve(population, num_evolutions=NUM_EVOLUTIONS)
