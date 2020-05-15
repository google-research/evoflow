_INTX = 'int32'
_FLOATX = 'float32'
_BACKEND = None


def floatx():
    """Returns the default float type, as a string.

    E.g. `'float16'`, `'float32'`, `'float64'`.

    Returns:
        str: the current default float type.

    Example:
    >>> B.floatx()
    'float32'
    """
    return _FLOATX


def set_floatx(value):
    """Sets the default float type.

    Note: It is not recommended to set this to float16 for training,
    as this will likely cause numeric stability issues

    Args:
        value (str): `'float16'`, `'float32'`, or `'float64'`.

    Example:
    >>> evoflow.backend.floatx()
    'float32'
    >>> evoflow.backend.set_floatx('float64')
    >>> evoflow.backend.floatx()
    'float64'
    >>> evoflow.backend.set_floatx('float32')
    Raises:
        ValueError: In case of invalid value.
    """
    global _FLOATX
    if value not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(value))
    _FLOATX = str(value)


def intx():
    """Returns the default int type, as a string.

    E.g. `'int8'`, `'unit8'`, `'int32'`.

    Returns:
        str: the current default int type.

    Example:
    >>> B.intx()
    'int32'
    """
    return _INTX


def set_intx(value):
    """Sets the default int type.

    Args:
        value (str): default int type to use.
        One of `{'int8', 'uint8', 'int16', 'uint16', 'int32', 'int64'}`

    Example:
    >>> evoflow.backend.intx()
    'int32'
    >>> evoflow.set_intx('uint8')
    >>> evoflow.backend.intx()
    'uint8'
    >>> evoflow.intx('float132')
    Raises:
        ValueError: In case of invalid value.
    """
    global _INTX
    if value not in {'int8', 'uint8', 'int16', 'uint16', 'int32', 'int64'}:
        raise ValueError('Unknown intx type: ' + str(value))
    _INTX = str(value)


def get_backend():
    "Return the backend used"
    return _BACKEND


def set_backend(name):
    """Set Geneflow backend to be a given framework

    Args:
        name(str): Name of the backend. {cupy, numpy, tensorflow}. Default
        to tensorflow.

    See:
        `load_backend.py` for the actual loading code.
    """
    global _BACKEND
    if name not in {'cupy', 'numpy', 'tensorflow'}:
        raise ValueError('Unknown backend: ' + str(name))

    _BACKEND = name
