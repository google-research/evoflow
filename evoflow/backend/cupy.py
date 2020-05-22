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

import cupy as cp
# sometime numpy is faster / cupy don't have the op.
import numpy  # DON'T IMPORT AS np -- make it hard to distinguish with cupy.

from .common import _infer_dtype, intx, floatx


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

    # cupy dont' support tuple
    if isinstance(a, tuple):
        a = list(a)

    if not dtype and not isinstance(a, numpy.ndarray):
        # automatic inference based of floatx and intx settings
        dtype = _infer_dtype(a)

    return cp.array(a, dtype)


def copy(tensor):
    """Copy a tensor

    Args:
        tensor (ndarray): tensor to copy
    Returns
        ndarray: copied tensor
    """
    return cp.copy(tensor)


def zeros(shape, dtype=floatx()):
    """Returns a new Tensor of given shape and dtype, filled with zeros.
    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype (str, optional): dtype: Data type specifier. Defaults to 'l'.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return cp.zeros(shape, dtype=dtype)


def ones(shape, dtype=floatx()):
    """Returns a new Tensor of given shape and dtype, filled with ones.
    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return cp.ones(shape, dtype=dtype)


def fill(shape, fill_value):
    """Returns a new Tensor of given shape and dtype, filled with the provided
    value.

    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        fill_value (int): The value to fill with..

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return cp.full(shape, fill_value)


def normal(shape, mean=0.0, dev=1.0):
    """Draw random samples from a normal (Gaussian) distribution.

    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        mean (float): Mean value. Default to 0.0.
        dev (float): Standard deviations. Default to 1.0.

    Returns:
        ndarray: Drawn samples from the parameterized normal distribution.

    """
    return cp.random.normal(loc=mean, scale=dev, size=shape)


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
    return cp.arange(start, stop=stop, step=step, dtype=dtype)


# - Reduce -
def prod(tensor, axis=None, keepdims=False):
    """Returns the product of an array along a given axis.

    Args:
        tensor (ndarray): Array to take the maximum.
        axis (int): Along which axis to take the maximum. The flattened array
            is used by default. Defaults to None.
        dtype: Data type specifier.
        keepdims (bool): If ``True``, the axis is kept as an axis of
        size one. Default to False.

    Returns:
        ndarray: The maximum of ``tensor``, along the axis if specified.
    """
    return cp.prod(tensor, axis=axis, keepdims=keepdims)


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

    # cupy don't support keepdims.
    if keepdims:
        return numpy.amax(tensor, axis=axis, keepdims=keepdims)
    else:
        return cp.amax(tensor, axis=axis, keepdims=keepdims)


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

    # cupy don't support keepdims.
    if keepdims:
        return numpy.amin(tensor, axis=axis, keepdims=keepdims)
    else:
        return cp.amin(tensor, axis=axis, keepdims=keepdims)


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
    return cp.sum(tensor, axis=axis, keepdims=keepdims)


def mean(tensor, axis=None, keepdims=False):
    """Returns the sum of an array along given axes.

    Args:
        tensor (ndarray): Array to mean reduce.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        keepdims (bool): If ``True``, the specified axes are remained as axes
        of length one.

    Returns:
        ndarray: The mean of ``tensor``, along the axis if specified.
    """
    return cp.mean(tensor, axis=axis, keepdims=keepdims)


def sqrt(tensor):
    """Computes element-wise square root of the input tensor.

    Args:
        tensor (ndarray): tensor

    Returns:
        tensor: square root of the input tensor.
    """
    return cp.sqrt(tensor)


# - Manipulation -


def reverse(tensor, axis):
    """Reverses specific dimensions of a tensor.

    Args:
        tensor (tensor): tensor to reverse
        axis (tensor): axis or tuple of axis
    """
    if isinstance(axis, int):
        return cp.flip(tensor, axis)
    else:
        for ax in axis:
            tensor = cp.flip(tensor, axis=ax)
        return tensor


def roll(tensor, shift, axis):
    """Rolls the elements of a tensor along an axis.

    Args:
        tensor (tensor): tensor to roll
        shift (tensor): offset to roll
        axis (tensor): axis to shift by

    Returns:
        [type]: [description]
    """
    return cp.roll(tensor, shift, axis)


def assign(dst_tensor, values, slices):
    """ assign values in tensor at the position specified by the slices.

    Args:
        dst_tensor (ndarray): Target tensor
        values (ndarray): Tensor containing the values to assign.
        slices (tuple): slices that defines where to put the values.
    Returns:
        ndarray: tensor with the assigned values.

    """
    dst_tensor[slices] = values
    return dst_tensor


def tile(tensor, reps):
    """Construct a tensor by repeating tensor the number of times given by reps.
    Args:
        tensor (cupy.ndarray): Tensor to transform.
        reps (int or tuple): The number of repeats.
    Returns:
        ndarray: Transformed tensor with repeats.
    """
    return cp.tile(tensor, reps)


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
    return cp.concatenate(tup, axis=axis)


# - Utils -


def transpose(a):
    "Transpose the tensor"
    return cp.transpose(a)


def cast(tensor, dtype):
    """Cast

    Args:
        tensor (Tensor): tensor to cast.
        dtype (str): type to cast. Usually floatx or intx
    """

    if isinstance(tensor, float) or isinstance(tensor, int):
        if dtype in ['float', 'float32']:
            return float(tensor)
        elif dtype == ['int', 'int32']:
            return int(tensor)
        else:
            raise ValueError("can't cast scalar", type(tensor), "to dype",
                             dtype)
    return tensor.astype(dtype)


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
    return tensor.flatten()


def as_numpy_array(tensor):
    """Convert tensor to a numpy array.

    Useful when interfacing with other librairies to have a unified input
    to them.

    Args:
        tensor (ndarray): tensor to convert
    Returns:
        numpy.ndarray: Tensor as numpy array
    """
    return cp.asnumpy(tensor)


def reshape(tensor, shape):
    "reshape tensor"
    return cp.reshape(tensor, shape)


def is_tensor(a):
    "check if the given object is a tensor"
    return isinstance(a, cp.ndarray)


def tensor_equal(tensor1, tensor2):
    """True if two tensors are exactly equals

    Args:
        tensor1 (ndarray): tensor to compare.
        tensor2 (ndarray): tensor to compare.

    Returns:
        Bool: True if exactly equal, False otherwise.

    """
    return (tensor1 == tensor2).all()


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

    return cp.allclose(a, b, atol=absolute_tolerance, rtol=relative_tolerance)


# - Math -
def dot(t1, t2):
    """Return the dot product of two arrays

    Args:
        t1 (ndarray): Left tensor
        t2 (ndarray): Right tensor

    Return:
        ndarray: tensor containing the dot product
    """
    return cp.dot(t1, t2)


def add(tensor1, tensor2):
    """Add two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
    """
    return cp.add(tensor1, tensor2)


def subtract(tensor1, tensor2, dtype=None):
    """Substract two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
    """
    return cp.subtract(tensor1, tensor2)


def multiply(tensor1, tensor2):
    """multiply two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
    """
    return cp.multiply(tensor1, tensor2)


def divide(numerator, denominator):
    """divide a tensor by another

    Args:
        tensor1 (ndarray): numerator tensor.
        tensor2 (ndarray): denominator tensor.
    """
    return cp.divide(numerator, denominator)


def mod(numerator, denominator):
    """Compute the reminder of the divisin of a tensor by another

    Args:
        tensor1 (ndarray): numerator tensor.
        tensor2 (ndarray): denominator tensor..
    """
    return cp.mod(numerator, denominator)


def clip(tensor, min_val=0, max_val=None):
    """Clips the values of a tensor to a given interval. For example,
    if an interval of [0, 1] is specified, values smaller than 0 become 0,
    and values larger than 1 become 1.

    Efficient version of ``max(min(a, max_val), min_val)``

    Args:
        tensor (ndarray): The input Tensor.

        min_val (scalar, ndarray): The left side of the interval.
        Defaults to 0.

        max_val (scalar, ndarray or None): The right side of the interval. When
        None ignored.  Defaults to None.

    Returns:
        ndarray: Clipped tensor.
    """
    return cp.clip(tensor, a_min=min_val, a_max=max_val)


def abs(tensor):
    "Calculate the absolute value element-wise."
    return cp.absolute(tensor)


def broadcasted_norm(tensor):
    "Norm broadcasted accross dimensions"
    return cp.sum(cp.abs(tensor)**2, axis=-1)**0.5


def norm(tensor, ord=None, axis=None, keepdims=False):
    """Return one of eight different matrix norms, or one of an infinite
    number of vector norms (described below), depending on the value of
    the `ord` parameter.

    Args:
        tensor (ndarray): Array to take the norm from. If ``axis`` is None,
        the ``tensor`` must be 1D or 2D.

        ord (non-zero int, inf, -inf, 'fro'): Norm type.
        See `numpy.linalg.norm` for explanation.

        axis (int, 2-tuple of ints, None): `axis` along which the norm is
        computed.

        keepdims (bool): If this is set ``True``, the axes which are normed
        over are left in the resulting tensor with a size of one.

    Returns:
        ndarray: Norm of the tensor.
    """
    return cp.linalg.norm(tensor, ord=ord, axis=axis, keepdims=keepdims)


# - Randomness -


def randint(low, high=None, shape=None, dtype='l'):
    """Returns a scalar or an array of integer values over [low, high)

    Args:
        low (int): If high is None, it is the upper bound of the
        interval and the lower bound is set to 0. if high is set, it is the
        lower bound of the interval.

        high (int, optional):Upper bound of the interval. Defaults to None.

        shape (None or int or tuple of ints, optional): The shape of returned
        value. Defaults to None.

        dtype (str, optional): dtype: Data type specifier. Defaults to 'l'.

    Returns:
        int or ndarray of ints: If size is None, it is single integer
        sampled. If size is integer, it is the 1D-array of length size
        element. Otherwise, it is the array whose shape specified by size.
    """
    return cp.random.randint(low, high=high, size=shape, dtype=dtype)


def shuffle(tensor, axis=0):
    """Shuffle in place a tensor along a given axis. Other axis remain in
    place.

    Args:
        tensor (ndarray): tensor to shuffle.
        axis (int, optional): axis to shuffle on. Default to 0.

    Returns:
        Tensor: shuffled tensor
    """
    # ! we must return the tensor because other backend don't do shuffle
    # ! in place.

    if not axis:
        cp.random.shuffle(tensor)
    else:
        size = tensor.shape[axis]
        # cupy don't support new numpy rng system, so have to do it ourselves.
        cp.take(tensor, cp.random.rand(size).argsort(), axis=axis, out=tensor)
    return tensor

    # alternative version
    # cp.take(tensor, cp.random.permutation(size), axis=axis, out=tensor)


def full_shuffle(tensor):
    """Shuffle in place a tensor along all of its axis

    Args:
        tensor (ndarray): tensor to shuffle.

    Returns:
        Tensor: shuffled tensor

    """
    # ! we must return the tensor because other backend don't do shuffle
    # ! in place.

    for idx in range(len(tensor.shape)):
        shuffle(tensor, axis=int(idx))

    return tensor


# - Indexing -


def take(t, indices, axis=None, out=None):
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
    if not is_tensor(t):
        t = tensor(t)
    return cp.take(t, indices, axis=0)


def top_k_indices(tensor, k):
    """
    Finds the indices of the k largest entries alongside an axis.

    Args:
        tensor (ndarray): Tensor with a last dimension of at least k size.

        k (i): number of elements to return.

    """
    k = -k  # reverse to get top elements

    # we do partition and then sort to maximizes speed and have an avg
    # complexity of O(k log k).
    idxs = cp.argpartition(tensor, k)[k:]

    # ! mind the - argsort return in increasing order we want largest
    return idxs[cp.argsort(-tensor[idxs])]


def bottom_k_indices(tensor, k, axis=-1):
    """
    Finds the indices of the k smallest entries alongside an axis.

    Args:
        tensor (ndarray): Tensor with a last dimension of at least k size.

        k (i): number of elements to return.

    """
    idxs = cp.argpartition(tensor, k)[:k]
    return idxs[cp.argsort(tensor[idxs])]


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
    return cp.unique(tensor, return_index=True, return_counts=True)


# - Statistical -


def bincount(tensor, weights=None, minlength=None):
    """Count number of occurrences of each value in array of non-negative ints.

    Args:
        tensor (ndarray): Input tensor.

        weights (cupy.ndarray): Weights tensor which has the same shape as
        tensor``. Default to None.

        minlength (int): A minimum number of bins for the output array.

    Note:
        cupy is significantly slower than numpy for bincount so we rely
        on numpy for this op. Benchmark number:

            -=[Timing counters]=-
            +--------------------------+----------+
            | name                     |    value |
            |--------------------------+----------|
            | bincount cp with weights | 4.44327  |
            | bincount cp              | 4.40655  |
            | bincount np with weights | 0.187499 |
            | bincount np              | 0.176528 |
            +--------------------------+----------+

        see benchmark notebook for additional benchmark or rerun this one.

    Returns:
        ndarray: The result of binning the input tensor. The length of
        output is equal to ``max(max(x) + 1, minlength)``.

    """
    return numpy.bincount(tensor, weights=weights, minlength=minlength)
