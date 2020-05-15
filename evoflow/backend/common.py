from evoflow.config import floatx, intx


def _infer_dtype(a):
    "infers what tensor.dtype to use for a given variable during conversion"
    if isinstance(a, int):
        dtype = intx()
    elif isinstance(a, float):
        dtype = floatx()
    elif isinstance(a, list):
        have_float = False
        for v in a:
            if isinstance(v, float):
                have_float = True
        if have_float:
            dtype = floatx()
        else:
            dtype = intx()
    else:
        raise ValueError("can't cast type:", type(a), 'to a tensor')
    return dtype
