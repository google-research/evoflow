from evoflow.backend import set_floatx, floatx, set_intx, intx


def test_set_floatx():
    for dtype in ['float16', 'float32', 'float64']:
        set_floatx(dtype)
        assert floatx() == dtype
        set_floatx('float32')


def test_set_intx():
    for dtype in ['int8', 'uint8', 'int16', 'uint16', 'int32', 'int64']:
        set_intx(dtype)
        assert intx() == dtype
        set_intx('int32')
