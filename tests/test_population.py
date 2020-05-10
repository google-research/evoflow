from geneflow.population import genRandIntPopulation
import geneflow.backend as B


def test_randintpop():
    shape = (100, 100, 10)
    pop = genRandIntPopulation(shape, 42, 1)
    assert pop.shape == shape
    assert B.max(pop) == 42
    assert B.min(pop) == 1
