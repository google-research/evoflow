from geneflow.fitness import Max
import geneflow.backend as B


def test_max():
    t = B.randint(0, 10, (10, 10, 10))
    v = Max().call(t)
    print(v[0])
    assert v.shape == (10, 10)
    assert B.sum(v[0] - B.sum(t[0], axis=-1)) == 0


def test_max_gene_val():
    t = B.randint(0, 10, (10, 10, 10))
    v = Max(expected_max_value=5).call(t)
    assert v.shape == (10, 10)
