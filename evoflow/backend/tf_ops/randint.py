import tensorflow as tf

RAND_GENERATOR = tf.random.get_global_generator()


@tf.function()
def randint2(low, high=None, shape=None, dtype='int32'):
    """Returns a scalar or an array of integer values over [low, high)

    Args:
        low (int): If high is None, it is the upper bound of the
        interval and the lower bound is set to 0. if high is set, it is the
        lower bound of the interval.

        high (int, optional):Upper bound of the interval. Defaults to None.

        shape (None or int or tuple of ints, optional): The shape of returned
        value. Defaults to None.

        dtype (str, optional): dtype: Data type specifier.
        Defaults to 'float32'.

    Returns:
        int or ndarray of ints: If size is None, it is single integer
        sampled. If size is integer, it is the 1D-array of length size
        element. Otherwise, it is the array whose shape specified by size.
    """

    # just one number
    if not shape:
        return RAND_GENERATOR.uniform(shape=(1, ),
                                      minval=low,
                                      maxval=high,
                                      dtype='int32')[0]

    if isinstance(shape, int):
        shape = (shape, )
    return RAND_GENERATOR.uniform(shape=shape,
                                  minval=low,
                                  maxval=high,
                                  dtype=dtype)
