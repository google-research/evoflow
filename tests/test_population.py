from evoflow.population import randint_population, uniform_population
import evoflow.backend as B


def test_randintpop():
    shape = (100, 100, 10)
    pop = randint_population(shape, 42, 1)
    assert pop.shape == shape
    assert B.max(pop) == 42
    assert B.min(pop) == 1


def test_uniformpop():
    shape = (100, 100)
    pop = uniform_population(shape)
    assert pop.shape == shape
    assert B.max(pop) == 99
    assert B.min(pop) == 0

    for chrm in pop:
        _, _, count = B.unique_with_counts(chrm)
        assert B.max(count) == 1
