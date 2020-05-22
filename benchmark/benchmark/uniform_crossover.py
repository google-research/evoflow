from evoflow.ops import UniformCrossover1D, UniformCrossover2D
from evoflow.ops import UniformCrossover3D


def bench_uniform_crossover(population):

    # setup
    shape = population.shape
    population_fraction = 1
    # select the right op
    if len(shape) == 2:
        mutations_probability = 0.5
        OP = UniformCrossover1D
    elif len(shape) == 3:
        mutations_probability = (0.5, 0.5)
        OP = UniformCrossover2D
    elif len(shape) == 4:
        mutations_probability = (0.5, 0.5, 0.5)
        OP = UniformCrossover3D
    else:
        raise ValueError("too many dimensions")

    OP(population_fraction=population_fraction,
       mutations_probability=mutations_probability)(population)
