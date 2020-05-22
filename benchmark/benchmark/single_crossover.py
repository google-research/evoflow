from evoflow.ops import SingleCrossover1D, SingleCrossover2D
from evoflow.ops import SingleCrossover3D


def bench_single_crossover(population):

    # setup
    shape = population.shape
    population_fraction = 1
    # select the right op
    if len(shape) == 2:
        mutations_probability = 0.5
        OP = SingleCrossover1D
    elif len(shape) == 3:
        mutations_probability = (0.5, 0.5)
        OP = SingleCrossover2D
    elif len(shape) == 4:
        mutations_probability = (0.5, 0.5, 0.5)
        OP = SingleCrossover3D
    else:
        raise ValueError("too many dimensions")

    OP(population_fraction=population_fraction,
       mutations_probability=mutations_probability)(population)
