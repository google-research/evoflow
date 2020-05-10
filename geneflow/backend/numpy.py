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

import numpy as np

# -initialization-


def copy(tensor):
    """Copy a tensor

    Args:
        tensor (ndarray): tensor to copy
    Returns
        ndarray: copied tensor
    """
    return np.copy(tensor)


def zeros(shape, dtype=float):
    """Returns a new Tensor of given shape and dtype, filled with zeros.
    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return np.zeros(shape, dtype=dtype)


def ones(shape, dtype=float):
    """Returns a new Tensor of given shape and dtype, filled with ones.
    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return np.ones(shape, dtype=dtype)


def full(shape, fill_value, dtype=float):
    """Returns a new Tensor of given shape and dtype, filled with the provided
    value.

    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        fill_value: The value to fill with.
        dtype: Data type specifier.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return np.full(shape, fill_value, dtype)


def normal(shape, mean=0.0, dev=1.0):
    """Draw random samples from a normal (Gaussian) distribution.
    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        mean: Mean values.
        dev: Standard deviations.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return np.random.normal(loc=mean, scale=dev, size=shape)


# - Reduce -
def prod(tensor, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the product of an array along a given axis.

    Args:
        tensor (ndarray): Array to take the maximum.
        axis (int): Along which axis to take the maximum. The flattened array
            is used by default. Defaults to None.
        dtype: Data type specifier.
        out (ndarray): Output array. Default to None.
        keepdims (bool): If ``True``, the axis is kept as an axis of
        size one. Default to False.

    Returns:
        ndarray: The maximum of ``tensor``, along the axis if specified.
    """
    return np.prod(tensor, axis=axis, out=out, keepdims=keepdims, dtype=dtype)


def max(tensor, axis=None, out=None, keepdims=False):
    """Returns the maximum of an array or the maximum along an axis.

    Note::
       When at least one element is NaN, the corresponding min value will be
       NaN.

    Args:
        tensor (ndarray): Array to take the maximum.
        axis (int): Along which axis to take the maximum. The flattened array
            is used by default. Defaults to None.
        dtype: Data type specifier.
        out (ndarray): Output array. Default to None.

        keepdims (bool): If ``True``, the axis is kept as an axis of
        size one. Default to False.

    Returns:
        ndarray: The maximum of ``tensor``, along the axis if specified.
    """
    return np.amax(tensor, axis=axis, out=out, keepdims=keepdims)


def min(tensor, axis=None, out=None, keepdims=False):
    """Returns the minimum of an array or the maximum along an axis.

    Note::
       When at least one element is NaN, the corresponding min value will be
       NaN.

    Args:
        tensor (ndarray): Array to take the maximum.
        axis (int): Along which axis to take the maximum. The flattened array
            is used by default. Defaults to None.
        out (ndarray): Output array. Default to None.

        keepdims (bool): If ``True``, the axis is kept as an axis of
        size one. Default to False.

    Returns:
        ndarray: The maximum of ``tensor``, along the axis if specified.
    """

    return np.amin(tensor, axis=axis, out=out, keepdims=keepdims)


def sum(tensor, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the sum of an array along given axes.

    Args:
        tensor (ndarray): Array to sum reduce.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
        of length one.

    Returns:
        ndarray: The sum of ``tensor``, along the axis if specified.
    """
    return np.sum(tensor, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def mean(tensor, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the sum of an array along given axes.

    Args:
        tensor (ndarray): Array to mean reduce.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
        of length one.

    Returns:
        ndarray: The mean of ``tensor``, along the axis if specified.
    """
    return np.mean(tensor, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


# - Manipulation -
def tensor(a, dtype=None):
    """Converts an object to a tensor.

    Args:
        a: The source object.
        dtype: Data type specifier. It is inferred from the input by default.
    Returns:
        ndarray: An array on the current device. If ``a`` is already on
        the device, no copy is performed.

    """
    return np.asarray(a, dtype)


def tile(tensor, reps):
    """Construct a tensor by repeating tensor the number of times given by reps.
    Args:
        tensor (cupy.ndarray): Tensor to transform.
        reps (int or tuple): The number of repeats.
    Returns:
        ndarray: Transformed tensor with repeats.
    """
    return np.tile(tensor, reps)


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
    return np.concatenate(tup, axis=axis)


# - Utils -
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
    return np.asarray(tensor)


def reshape(tensor, shape):
    "reshape tensor"
    return np.reshape(tensor, shape)


def is_tensor(a):
    "check if the given object is a tensor"
    return isinstance(a, np.ndarray)


def tensor_equal(tensor1, tensor2):
    """True if two tensors are exactly equals

    Args:
        tensor1 (ndarray): tensor to compare.
        tensor2 (ndarray): tensor to compare.

    Returns:
        Bool: True if exactly equal, False otherwise.

    """
    return np.array_equal(tensor1, tensor2)


def allclose(a, b, absolute_tolerance=0.1):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    Args:
        a (ndarray): Tensor with a last dimension of at least k size.
        b (ndarray): Tensor with a last dimension of at least k size.
        absolute_tolerance: Defined as abs(a[i]-b[i]) < absolute_tolerance.

    """
    return np.allclose(a, b, atol=absolute_tolerance, rtol=0)


# - Math -


def dot(t1, t2):
    """Return the dot product of two arrays

    Args:
        t1 (ndarray): Left tensor
        t2 (ndarray): Right tensor

    Return:
        ndarray: tensor containing the dot product
    """
    return np.dot(t1, t2)


def add(tensor1, tensor2, dtype=None):
    """Add two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
        dtype (dtype): type of the returned tensor
    """
    return np.add(tensor1, tensor2, dtype=dtype)


def subtract(tensor1, tensor2, dtype=None):
    """Substract two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
        dtype (dtype): type of the returned tensor.
    """
    return np.subtract(tensor1, tensor2, dtype=dtype)


def multiply(tensor1, tensor2, dtype=None):
    """multiply two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
        dtype (dtype): type of the returned tensor.
    """
    return np.multiply(tensor1, tensor2, dtype=dtype)


def divide(numerator, denominator, dtype=None):
    """divide a tensor by another

    Args:
        tensor1 (ndarray): numerator tensor.
        tensor2 (ndarray): denominator tensor.
        dtype (dtype): type of the returned tensor.
    """
    return np.divide(numerator, denominator, dtype=dtype)


def mod(numerator, denominator, dtype=None):
    """Compute the reminder of the divisin of a tensor by another

    Args:
        tensor1 (ndarray): numerator tensor.
        tensor2 (ndarray): denominator tensor.
        dtype (dtype): type of the returned tensor.
    """
    return np.mod(numerator, denominator, dtype=dtype)


def clip(tensor, min_val=None, max_val=None, out=None):
    """Clips the values of a tensor to a given interval. For example,
    if an interval of [0, 1] is specified, values smaller than 0 become 0,
    and values larger than 1 become 1.

    Efficient version of ``max(min(a, max_val), min_val)``

    Args:
        tensor (ndarray): The input Tensor.

        min_val (scalar, ndarray or None): The left side of the interval. When
        None ignored. Defaults to None.

        max_val (scalar, ndarray or None): The right side of the interval. When
        None ignored. Defaults to None.

        out (ndarray): Output array. Mostly used for inplace.
    Returns:
        ndarray: Clipped tensor.
    """
    return np.clip(tensor, a_min=min_val, a_max=max_val, out=out)


def abs(tensor):
    "Calculate the absolute value element-wise."
    return np.absolute(tensor)


def absolute(tensor):
    "Calculate the absolute value element-wise."
    return np.absolute(tensor)


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
    return np.linalg.norm(tensor, ord=ord, axis=axis, keepdims=keepdims)


# - Randomness -


def randint(low, high=None, shape=None, dtype='l'):
    """Returns a scalar or an array of integer values over [low, high)

    Args:
        low (int): If high is not None, it is the lower bound of the
        interval. Otherwise, it is the upper bound of the interval and
        lower bound of the interval is set to 0.

        high (int, optional):Upper bound of the interval. Defaults to None.

        shape (None or int or tuple of ints, optional): The shape of returned
        value. Defaults to None.

        dtype (str, optional): dtype: Data type specifier. Defaults to 'l'.

    Returns:
        int or ndarray of ints: If size is None, it is single integer
        sampled. If size is integer, it is the 1D-array of length size
        element. Otherwise, it is the array whose shape specified by size.
    """
    return np.random.randint(low, high=high, size=shape, dtype=dtype)


def shuffle(tensor, axis=0):
    """Shuffle in place a tensor along a given axis. Other axis remain
    in place.


    Args:
        tensor (ndarray): tensor to shuffle.
        axis (int, optional): axis to shuffle on. Default to 0.

    Returns:
        None: in place shuffling
    """
    rng = np.random.default_rng()
    rng.shuffle(tensor, axis=axis)


def full_shuffle(tensor):
    """Shuffle in place a tensor along all of its axis

    Args:
        tensor (ndarray): tensor to shuffle.

    Returns:
        None: in place shuffling

    """

    for idx in range(len(tensor.shape)):
        shuffle(tensor, axis=idx)


# - Indexing -


def take(tensor, indices, axis=None, out=None):
    """Takes elements of a Tensor at specified indices along a specified axis

    Args:
        tensor (ndarray): Tensor to extract elements from.

        indices (int or array-like): Indices of elements that this function
        takes.

        axis (int, optional): The axis along which to select indices from.
        The flattened input is used by default. Defaults to None.

        out (ndarray. Optional): Output array. If provided, it should be of
            appropriate shape and dtype. Defaults to None.
    Returns:
        ndarray: Tensor containing the values from the specified indices.
    """
    return np.take(tensor, indices, axis=axis, out=out)


def top_k_indices(tensor, k, axis=-1):
    """
    Finds the indices of the k largest entries alongside an axis.

    Args:
        tensor (ndarray): Tensor with a last dimension of at least k size.

        k (i): number of elements to return.

        axis (int or None) - Axis along which to sort. Default is -1,
        which is the last axis. If None is supplied,
        the array will be flattened before sorting.

    """
    k = -k  # reverse to get top elements

    # we do partition and then sort to maximizes speed and have an avg
    # complexity of O(k log k).
    idxs = np.argpartition(tensor, k)[k:]

    # ! mind the - argsort return in increasing order we want largest
    return idxs[np.argsort(-tensor[idxs])]


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
    idxs = np.argpartition(tensor, k)[:k]
    return idxs[np.argsort(tensor[idxs])]


# - Statistical -


def bincount(tensor, weights=None, minlength=0):
    """Count number of occurrences of each value in array of non-negative ints.

    Args:
        tensor (ndarray): Input tensor.

        weights (cupy.ndarray): Weights tensor which has the same shape as
        tensor``. Default to None.

        minlength (int): A minimum number of bins for the output array.

    Returns:
        ndarray: The result of binning the input tensor. The length of
        output is equal to ``max(max(x) + 1, minlength)``.

    """
    return np.bincount(tensor, weights=weights, minlength=minlength)
