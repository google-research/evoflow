from evoflow.engine import EvoFlow
from evoflow.ops import Input, RandomMutations1D, Shuffle, Reverse1D


def test_shape():
    SHAPE = (100, 100)

    inputs = Input(SHAPE)
    x = inputs
    x = Reverse1D()(x)
    x = Shuffle(population_fraction=0.1)(x)
    outputs = RandomMutations1D()(x)
    ef = EvoFlow(inputs, outputs, debug=0)

    for op_idx in ef.execution_path:
        op = ef.idx2op[op_idx]
        shape = op.get_output_shapes()
        assert len(shape) == 2
        assert shape[0] == 100
    ef.summary()