# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.errors import InvalidArgumentError

from .common import _infer_dtype
from evoflow.config import intx, floatx
import numpy


# -initialization-
def tensor(a, dtype=None):
    """Converts an object to a tensor.

    Args:
        a: The source object.
        dtype: Data type specifier. It is inferred from the input by default.
    Returns:
        ndarray: An array on the current device. If ``a`` is already on
        the device, no copy is performed.

    """
    if is_tensor(a):
        return a

    if not dtype and not isinstance(a, numpy.ndarray):
        # automatic inference based of floatx and intx settings
        dtype = _infer_dtype(a)

    return tf.convert_to_tensor(a, dtype=dtype)


def copy(tensor):
    """Copy a tensor

    Args:
        tensor (ndarray): tensor to copy
    Returns
        ndarray: copied tensor
    """
    return tf.identity(tensor)


def zeros(shape, dtype=float):
    """Returns a new Tensor of given shape and dtype, filled with zeros.
    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype (str, optional): dtype: Data type specifier. Defaults to 'l'.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return tf.zeros(shape, dtype=dtype)


def ones(shape, dtype=float):
    """Returns a new Tensor of given shape and dtype, filled with ones.
    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return tf.ones(shape, dtype=dtype)


def fill(shape, fill_value):
    """Returns a new Tensor of given shape and dtype, filled with the provided
    value.

    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        fill_value (int): The value to fill with.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return tf.fill(shape, fill_value)


def normal(shape, mean=0, stddev=1.0):
    """Draw random samples from a normal (Gaussian) distribution.

    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        mean (float): Mean value. Default to 0.0.
        stddev (float): Standard deviations. Default to 1.0.

    Returns:
        ndarray: Drawn samples from the parameterized normal distribution.

    """
    return tf.random.normal(shape, mean=mean, stddev=stddev)


def range(start, stop=None, step=1, dtype=intx()):
    """Creates a sequence of numbers.
    Creates a sequence of numbers that begins at `start` and extends by
    increments of `delta` up to but not including `stop`.

    Args:
        start (int): Initial value. Optional. Defaults to 0
        stop (int, optional): End value.
        delta (int, optional): Spacing between values.  Defaults to 1.
        dtype (str, optional): Tensor tyoe. Defaults to intx().

    Returns:
        Tensor: Tensor that contains the requested range.
    """
    return tf.range(start, limit=stop, delta=step, dtype=dtype)


# - Reduce -
def prod(tensor, axis=None, keepdims=False):
    """Returns the product of an array along a given axis.

    Args:
        tensor (ndarray): Array to take the maximum.
        axis (int): Along which axis to take the maximum. The flattened array
            is used by default. Defaults to None.
        keepdims (bool): If ``True``, the axis is kept as an axis of
        size one. Default to False.

    Returns:
        ndarray: The maximum of ``tensor``, along the axis if specified.
    """
    return tf.math.reduce_prod(tensor, axis=axis, keepdims=keepdims)


def max(tensor, axis=None, keepdims=False):
    """Returns the maximum of an array or the maximum along a given axis.

    Note::
       When at least one element is NaN, the corresponding min value will be
       NaN.

    Args:
        tensor (ndarray): Array to take the maximum.
        axis (int): Along which axis to take the maximum. The flattened array
            is used by default. Defaults to None.
        keepdims (bool): If ``True``, the axis is kept as an axis of
        size one. Default to False.

    Returns:
        ndarray: The maximum of ``tensor``, along the axis if specified.
    """

    return tf.math.reduce_max(tensor, axis=axis, keepdims=keepdims)


def min(tensor, axis=None, keepdims=False):
    """Returns the minimum of an array or the maximum along an axis.

    Note::
       When at least one element is NaN, the corresponding min value will be
       NaN.

    Args:
        tensor (ndarray): Array to take the maximum.
        axis (int): Along which axis to take the maximum. The flattened array
                    is used by default. Defaults to None.
        keepdims (bool): If ``True``, the axis is kept as an axis of
        size one. Default to False.

    Returns:
        ndarray: The maximum of ``tensor``, along the axis if specified.
    """
    return tf.math.reduce_min(tensor, axis=axis, keepdims=keepdims)


def sum(tensor, axis=None, keepdims=False):
    """Returns the sum of an array along given axes.

    Args:
        tensor (ndarray): Array to sum reduce.
        axis (int or sequence of ints): Axes along which the sum is taken.
        keepdims (bool): If ``True``, the specified axes are remained as axes
        of length one.


    Returns:
        ndarray: The sum of ``tensor``, along the axis if specified.
    """
    return tf.math.reduce_sum(tensor, axis=axis, keepdims=keepdims)


def mean(tensor, axis=None, keepdims=False):
    """Returns the sum of an array along given axes.

    Args:
        tensor (ndarray): Array to mean reduce.
        axis (int or sequence of ints): Axes along which the sum is taken.
        keepdims (bool): If ``True``, the specified axes are remained as axes
        of length one.

    Returns:
        ndarray: The mean of ``tensor``, along the axis if specified.
    """
    return tf.math.reduce_mean(tensor, axis=axis, keepdims=keepdims)


def sqrt(tensor):
    """Computes element-wise square root of the input tensor.

    Args:
        tensor (ndarray): tensor

    Returns:
        tensor: square root of the input tensor.
    """
    return tf.math.sqrt(tensor)


# - Manipulation -
def reverse(tensor, axis):
    """Reverses specific dimensions of a tensor.

    Args:
        tensor (tensor): tensor to reverse
        axis (tensor): axis or tuple of axis
    """
    return tf.reverse(tensor, axis)


def roll(tensor, shift, axis):
    """Rolls the elements of a tensor along an axis.

    Args:
        tensor (tensor): tensor to roll
        shift (tensor): offset to roll
        axis (tensor): axis to shift by

    Returns:
        [type]: [description]
    """
    return tf.roll(tensor, shift, axis)


def assign(dst_tensor, values, slices):
    """ assign values in tensor at the position specified by the slices.

    Args:
        dst_tensor (ndarray): Target tensor
        values (ndarray): Tensor containing the values to assign.
        slices (tuple): slices that defines where to put the values.
    Returns:
        ndarray: tensor with the assigned values.

    """
    # FIXME: its a hack to get it to work, there must be a faster way.

    dst_tensor = as_numpy_array(dst_tensor)
    values = as_numpy_array(values)
    dst_tensor[slices] = values
    return tensor(dst_tensor)


def tile(tensor, reps):
    """Construct a tensor by repeating tensor the number of times given by reps.
    Args:
        tensor (cupy.ndarray): Tensor to transform.
        reps (int or tuple): The number of repeats.
    Returns:
        ndarray: Transformed tensor with repeats.
    """
    return tf.tile(tensor, reps)


def concatenate(tup, axis=0):
    """Joins tensors along a given axis.
    Args:
        tup (sequence of arrays): Tensors to be joined. Tensors must have the
        same cardinality except for the specified axis.

        axis (int or None): Axis to joint along. If None, tensors are
        flattened before use. Default is 0.
    Returns:
        ndarray: Joined array.
    """
    return tf.concat(tup, axis=axis)


# - Utils -
def transpose(a):
    "Transpose the tensor"
    return tf.transpose(a)


def cast(tensor, dtype):
    """Cast

    Args:
        tensor (Tensor): tensor to cast.
        dtype (str): type to cast. Usually floatx() or intx()

    Returns:
        ndarray: Tensor casted in the requested format.
    """
    return tf.cast(tensor, dtype)


def dtype(tensor):
    """"Returns the dtype of a tensor as a string.

    Args:
        tensor (tensor): Tensor

    Returns:
        str: type of the tensor as string. e.g int32.
    """
    return tensor.dtype.name


def flatten(tensor):
    """Returns a copy of the tensor flatten into one dimension.

    Args:
        tensor (ndarray): tensor to flatten
    Returns:
        ndarray: flattened tensor
    """
    # note: unsure if that is the fastest way. maybe compute flat_shape in
    # pure python

    flat_shape = prod(tensor.shape)
    return tf.reshape(tensor, flat_shape)


def as_numpy_array(t):
    """Convert tensor to a numpy array.

    Useful when interfacing with other librairies to have a unified input
    to them.

    Args:
        t (ndarray): tensor to convert
    Returns:
        numpy.ndarray: Tensor as numpy array
    """

    if not is_tensor(t):
        dtype = _infer_dtype(t)
        if isinstance(t, list):
            t = tensor(t)
        else:
            t = tensor([t])
    else:
        dtype = t.dtype.name

    return t.numpy().astype(dtype)


def reshape(tensor, shape):
    "reshape tensor"
    return tf.reshape(tensor, shape)


def is_tensor(a):
    "check if the given object is a tensor"
    if isinstance(a, tf.Tensor):
        return True
    elif isinstance(a, tf.python.framework.tensor_shape.TensorShape):
        return True
    else:
        return False


def tensor_equal(tensor1, tensor2):
    """True if two tensors are exactly equals

    Args:
        tensor1 (ndarray): tensor to compare.
        tensor2 (ndarray): tensor to compare.

    Returns:
        Bool: True if exactly equal, False otherwise.

    """
    return tf.reduce_all(tf.equal(tensor1, tensor2))


def assert_near(a, b, absolute_tolerance=0, relative_tolerance=0):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    Args:
        a (ndarray): Tensor with a last dimension of at least k size.
        b (ndarray): Tensor with a last dimension of at least k size.
        absolute_tolerance (float): Default to 0
        relative_tolerance (float): Default to 0

    Returns:
        bool: True if the two arrays are equal within the given tolerance; False otherwise.

    Note:
        This function return True if the following equation is satisfied:
        `absolute(a - b) <= (absolute_tolerance + relative_tolerance * absolute(b))`  # noqa

    """

    try:
        tf.debugging.assert_near(a,
                                 b,
                                 atol=absolute_tolerance,
                                 rtol=relative_tolerance)
    except InvalidArgumentError:
        return False
    return True


# - Math -
def _is_scalar(a):
    if isinstance(a, int):
        return True
    elif isinstance(a, float):
        return True
    elif is_tensor(a):
        if a.shape == ():
            return True
    else:
        False


def dot(t1, t2):
    """Return the dot product of two arrays

    Args:
        t1 (ndarray): Left tensor
        t2 (ndarray): Right tensor

    Return:
        ndarray: tensor containing the dot product
    """

    # # scalar
    if (_is_scalar(t1) or _is_scalar(t2)):
        return t1 * t2
    #     return tf.reduce_sum(tf.multiply(t1, t2))
    elif len(t1.shape) == 1 or len(t2.shape) == 1:
        return tf.reduce_sum(tf.multiply(t1, t2), axis=-1)
    else:
        return tf.matmul(t1, t2)


def add(tensor1, tensor2):
    """Add two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
    """
    return tf.add(tensor1, tensor2)


def subtract(tensor1, tensor2):
    """Substract two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
    """
    return tf.subtract(tensor1, tensor2)


def multiply(tensor1, tensor2):
    """multiply two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
    """
    return tf.multiply(tensor1, tensor2)


def divide(numerator, denominator):
    """divide a tensor by another

    Args:
        tensor1 (ndarray): numerator tensor.
        tensor2 (ndarray): denominator tensor.
    """
    return tf.cast(tf.divide(numerator, denominator), floatx())


def mod(numerator, denominator):
    """Compute the reminder of the divisin of a tensor by another

    Args:
        tensor1 (ndarray): numerator tensor.
        tensor2 (ndarray): denominator tensor.
    """
    return tf.math.mod(numerator, denominator)


def clip(tensor, min_val=0, max_val=None, out=None):
    """Clips the values of a tensor to a given interval. For example,
    if an interval of [0, 1] is specified, values smaller than 0 become 0,
    and values larger than 1 become 1.

    Efficient version of ``max(min(a, max_val), min_val)``

    Args:
        tensor (ndarray): The input Tensor.

        min_val (scalar, ndarray or None): The left side of the interval. When
        None ignored.  Defaults to None.

        max_val (scalar, ndarray or None): The right side of the interval. When
        None ignored.  Defaults to None.

    Returns:
        ndarray: Clipped tensor.
    """
    return tf.clip_by_value(tensor,
                            clip_value_min=min_val,
                            clip_value_max=max_val)


def abs(tensor):
    "Calculate the absolute value element-wise."
    return tf.math.abs(tensor)


def broadcasted_norm(tensor):
    "Norm broadcasted accross dimensions"
    norm = cast(tf.abs(tensor), intx())
    norm = norm**2
    norm = tf.reduce_sum(norm, axis=-1)
    norm = sqrt(cast(norm, floatx()))
    return norm

    # population_norm = B.sum(B.abs(flat_pop)**2, axis=-1)**0.5


def norm(tensor, ord='euclidean', axis=None, keepdims=False):
    """Return one of eight different matrix norms, or one of an infinite
    number of vector norms (described below), depending on the value of
    the `ord` parameter.

    Args:
        tensor (ndarray): Array to take the norm from. If ``axis`` is None,
        the ``tensor`` must be 1D or 2D.

        ord (non-zero int, inf, -inf, 'fro'): Norm type. Euclidian by default.
        See `tf.norm` for explanation:
        https://www.tensorflow.org/api_docs/python/tf/norm

        axis (int, 2-tuple of ints, None): `axis` along which the norm is
        computed.

        keepdims (bool): If this is set ``True``, the axes which are normed
        over are left in the resulting tensor with a size of one.

    Returns:
        ndarray: Norm of the tensor.
    """
    return tf.norm(tensor, ord=ord, axis=axis, keepdims=keepdims)


# - Randomness -


def randint(low, high=None, shape=None, dtype=intx()):
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
        return numpy.random.randint(low, high=high)

    if isinstance(shape, int):
        shape = (shape, )
    return tf.random.uniform(shape=shape, minval=low, maxval=high, dtype=dtype)


def shuffle(t, axis=0):
    """Shuffle tensor along the given axis. Other axis remain in
    place.

    Args:
        tensor (ndarray): tensor to shuffle.
        axis (int, optional): axis to shuffle on. Default to 0.

    Returns:
        None: in place shuffling
    """
    if not axis:
        # ! tensorflow don't do in place shuffling
        return tf.random.shuffle(t)
    else:
        # FIXME: its a hack as we use numpy which is liklely to cause slowness
        t = as_numpy_array(t)
        rng = numpy.random.default_rng()
        rng.shuffle(t, axis=axis)
        return tensor(t)


def full_shuffle(t):
    """Shuffle in place a tensor along all of its axis

    Args:
        t (ndarray): tensor to shuffle.


    Returns:
        None: in place shuffling

    """
    # ! dont use the variable name tensor as it confusion with the tensor()
    # ! function

    # FIXME: its a hack as we use numpy which is liklely to cause slowness
    t = as_numpy_array(t)
    rng = numpy.random.default_rng()
    for idx in range(len(t.shape)):
        rng.shuffle(t, axis=idx)
    return tensor(t)


# - Indexing -
def take(tensor, indices, axis=None):
    """Takes elements of a Tensor at specified indices along a specified axis

    Args:
        tensor (ndarray): Tensor to extract elements from.

        indices (int or array-like): Indices of elements that this function
        takes.

        axis (int, optional): The axis along which to select indices from.
        The flattened input is used by default. Defaults to None.

    Returns:
        ndarray: Tensor containing the values from the specified indices.
    """
    return tf.gather(tensor, indices, axis=axis)


def top_k_indices(tensor, k, axis=-1):
    """
    Finds the indices of the k largest entries alongside an axis.

    Args:
        tensor (ndarray): Tensor with a last dimension of at least k size.

        k (i): number of elements to return

    """
    return tf.math.top_k(tensor, k)[1]


def bottom_k_indices(tensor, k, axis=-1):
    """
    Finds the indices of the k smallest entries alongside an axis.

    Args:
        tensor (ndarray): Tensor with a last dimension of at least k size.

        k (i): number of elements to return.

        axis (int or None) - Axis along which to sort. Default is -1,
        which is the last axis. If None is supplied,
        the array will be flattened before sorting.

    """
    # inverted top k and reinverted before returning
    return tf.math.top_k(tf.multiply(tensor, -1), k)[1]


def unique_with_counts(tensor):
    """Finds unique elements and return them along side their position and
    counts.

    Args:
        tensor (Tensor): 1D tensor to analyze.

    Returns:
        values (Tensor): unique values founded
        indexes (Tensor): index of the value sorted
        counts (Tensor): Tensor containing the count for each value.
    """
    return tf.unique_with_counts(tensor)


# - Statistical -


def bincount(tensor, weights=None, minlength=None):
    """Count number of occurrences of each value in array of non-negative ints.

    Args:
        tensor (ndarray): Input tensor.

        weights (cupy.ndarray): Weights tensor which has the same shape as
        tensor``. Default to None.

        minlength (int): A minimum number of bins for the output array.

    """
    return tf.math.bincount(tensor, weights=weights, minlength=minlength)
