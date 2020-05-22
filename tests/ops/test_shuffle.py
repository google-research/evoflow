from evoflow.ops import Shuffle
from evoflow.population import randint_population
import evoflow.backend as B


def test_shuffle():

    SHAPES = [(2, 4), (2, 4, 4), (2, 4, 4, 4)]

    for shape in SHAPES:
        population = randint_population(shape, max_value=255)
        previous_population = B.copy(population)
        population = Shuffle(1, debug=True)(population)
        diff = B.clip(abs(population - previous_population), 0, 1)

        assert B.is_tensor(population)
        assert population.shape == shape
        assert B.sum(diff) > int(B.prod(shape) / 2)
