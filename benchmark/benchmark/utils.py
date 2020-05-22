def shape2opdim(shape):
    if len(shape) == 2:
        OP_DIM = "1D"
    elif len(shape) == 3:
        OP_DIM = "2D"
    elif len(shape) == 4:
        OP_DIM = "3D"
    else:
        raise ValueError('Too many dimensions')
    return OP_DIM
