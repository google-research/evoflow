from termcolor import cprint
import geneflow.backend as B
from geneflow.ops import UniformCrossover1D, UniformCrossover2D
from geneflow.ops import UniformCrossover3D


def test_crossover2d_shape():
    GENOME_SHAPE = (10, 4, 4)
    MAX_VAL = 250
    chromosomes = B.randint(0, MAX_VAL, GENOME_SHAPE)
    num_crossover_fraction = 0.5
    max_crossover_size_fraction = (0.5, 0.5)

    res = UniformCrossover2D(num_crossover_fraction,
                             max_crossover_size_fraction)(chromosomes)
    assert res.shape == GENOME_SHAPE
    assert B.max(res) <= MAX_VAL


def test_uniform_2Dcrossover_randomness_shape():
    GENOME_SHAPE = (10, 4, 4)
    chromosomes = B.randint(0, 1024, GENOME_SHAPE)
    num_crossover_fraction = 0.5
    max_crossover_size_fraction = (0.5, 0.5)

    res = UniformCrossover2D(num_crossover_fraction,
                             max_crossover_size_fraction,
                             debug=True)(chromosomes)

    diff = B.clip(abs(res['mutated'] - res['original']), 0, 1)
    expected_mutated = chromosomes.shape[0] * num_crossover_fraction

    mutated_chromosomes = []
    for c in diff:
        if B.max(c):
            mutated_chromosomes.append(c)

    # sometime we have a collision
    assert abs(len(mutated_chromosomes) - expected_mutated) < 2

    mutated_rows = max_crossover_size_fraction[0] * GENOME_SHAPE[1]
    mutated_cells = max_crossover_size_fraction[0] * GENOME_SHAPE[2]
    for cidx, c in enumerate(mutated_chromosomes):
        mr = 0
        mc = 0
        for r in c:
            s = B.sum(r)
            if s:
                mr += 1
                mc += s

        assert abs(mutated_rows - mr) < 2
        assert abs(mutated_cells - (mc // mutated_rows)) < 2


def test_uniform_2Dcrossover_distribution():
    "check that every gene of the tensor are going to be flipped"
    GENOME_SHAPE = (10, 4, 4)
    chromosomes = B.randint(0, 1024, GENOME_SHAPE)
    num_crossover_fraction = 0.5
    max_crossover_size_fraction = (0.5, 0.5)

    res = UniformCrossover2D(num_crossover_fraction,
                             max_crossover_size_fraction,
                             debug=True)(chromosomes)

    # diff matrix
    UC = UniformCrossover2D(num_crossover_fraction,
                            max_crossover_size_fraction,
                            debug=1)
    diff = B.clip(abs(res['mutated'] - res['original']), 0, 1)
    for _ in range(200):
        res = UC(chromosomes)
        # acumulating diff matrix
        diff += B.clip(abs(res['mutated'] - res['original']), 0, 1)

    # each gene proba of being mutated 1/2*1/2 > 1/4
    # each chromosome proba of being mutated 1/2
    # => gene average hit rate: 1000 / (1/4*1/2)  ~25
    for c in diff:
        print(c)
        print('mean', B.mean(c), 'min', B.min(c), 'max', B.max(c))
        assert B.min(c) > 10
        assert B.max(c) < 50
        assert 10 < B.mean(c) < 50
