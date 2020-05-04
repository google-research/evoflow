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

# -initialization-


def copy(tensor):
    """Copy a tensor

    Args:
        tensor (ndarray): tensor to copy
    Returns
        ndarray: copied tensor
    """
    return cp.copy(tensor)


def zeros(shape, dtype=float):
    """Returns a new Tensor of given shape and dtype, filled with zeros.
    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype (str, optional): dtype: Data type specifier. Defaults to 'l'.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return cp.zeros(shape, dtype=dtype)


def ones(shape, dtype=float):
    """Returns a new Tensor of given shape and dtype, filled with ones.
    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return cp.ones(shape, dtype=dtype)


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
    return cp.full(shape, fill_value, dtype)


def normal(shape, mean=0.0, dev=1.0):
    """Draw random samples from a normal (Gaussian) distribution.

    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        mean: Mean values.
        dev: Standard deviations.

    Returns:
        ndarray: An tensor filled with zeros.

    """
    return cp.random.normal(loc=mean, scale=dev, size=shape)


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
    return cp.prod(tensor, axis=axis, out=out, keepdims=keepdims, dtype=dtype)


def max(tensor, axis=None, out=None, keepdims=False):
    """Returns the maximum of an array or the maximum along a given axis.

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

    # cupy don't support keepdims.
    if keepdims:
        return numpy.amax(tensor, axis=axis, out=out, keepdims=keepdims)
    else:
        return cp.amax(tensor, axis=axis, out=out, keepdims=keepdims)


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

    # cupy don't support keepdims.
    if keepdims:
        return numpy.amin(tensor, axis=axis, out=out, keepdims=keepdims)
    else:
        return cp.amin(tensor, axis=axis, out=out, keepdims=keepdims)


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
    return cp.sum(tensor, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def mean(tensor, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the sum of an array along given axes.

    Args:
        tensor (ndarray): Array to mean reduce.
        axis (int or sequence of ints): Axes along which the sum is taken.
        out (cupy.ndarray): Output array.
        dtype: Data type specifier.
        keepdims (bool): If ``True``, the specified axes are remained as axes
        of length one.
        dtype: Data type specifier.

    Returns:
        ndarray: The mean of ``tensor``, along the axis if specified.
    """
    return cp.mean(tensor, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


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
    return cp.array(a, dtype)


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


def allclose(a, b, absolute_tolerance=0.1):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    Args:
        a (ndarray): Tensor with a last dimension of at least k size.
        b (ndarray): Tensor with a last dimension of at least k size.
        absolute_tolerance: Defined as abs(a[i]-b[i]) < absolute_tolerance.

    """
    return cp.allclose(a, b, atol=absolute_tolerance, rtol=0)


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


def add(tensor1, tensor2, dtype=None):
    """Add two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
        dtype (dtype): type of the returned tensor.
    """
    return cp.add(tensor1, tensor2, dtype=dtype)


def subtract(tensor1, tensor2, dtype=None):
    """Substract two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
        dtype (dtype): type of the returned tensor.
    """
    return cp.subtract(tensor1, tensor2, dtype=dtype)


def multiply(tensor1, tensor2, dtype=None):
    """multiply two tensors

    Args:
        tensor1 (ndarray): Left tensor.
        tensor2 (ndarray): right tensor.
        dtype (dtype): type of the returned tensor.
    """
    return cp.multiply(tensor1, tensor2, dtype=dtype)


def divide(numerator, denominator, dtype=None):
    """divide a tensor by another

    Args:
        tensor1 (ndarray): numerator tensor.
        tensor2 (ndarray): denominator tensor.
        dtype (dtype): type of the returned tensor.
    """
    return cp.divide(numerator, denominator, dtype=dtype)


def mod(numerator, denominator, dtype=None):
    """Compute the reminder of the divisin of a tensor by another

    Args:
        tensor1 (ndarray): numerator tensor.
        tensor2 (ndarray): denominator tensor.
        dtype (dtype): type of the returned tensor.
    """
    return cp.mod(numerator, denominator, dtype=dtype)


def clip(tensor, min_val=None, max_val=None, out=None):
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

        out (ndarray): Output array. Mostly used for inplace.
    Returns:
        ndarray: Clipped tensor.
    """
    return cp.clip(tensor, a_min=min_val, a_max=max_val, out=out)


def abs(tensor):
    "Calculate the absolute value element-wise."
    return cp.absolute(tensor)


def absolute(tensor):
    "Calculate the tensor norm fosrm."
    return cp.absolute(tensor)


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
    return cp.random.randint(low, high=high, size=shape, dtype=dtype)


def shuffle(tensor, axis=0):
    """Shuffle in place a tensor along a given axis. Other axis remain in
    place.

    Args:
        tensor (ndarray): tensor to shuffle.
        axis (int, optional): axis to shuffle on. Default to 0.

    Returns:
        None: in place shuffling
    """
    if not axis:
        cp.random.shuffle(tensor)
    else:
        size = tensor.shape[axis]
        # cupy don't support new numpy rng system, so have to do it ourselves.
        cp.take(tensor, cp.random.rand(size).argsort(), axis=axis, out=tensor)

        # alternative version
        # cp.take(tensor, cp.random.permutation(size), axis=axis, out=tensor)


def full_shuffle(tensor):
    """Shuffle in place a tensor along all of its axis

    Args:
        tensor (ndarray): tensor to shuffle.

    Returns:
        None: in place shuffling

    """

    for idx in range(len(tensor.shape)):
        shuffle(tensor, axis=idx)

    # alternative
    # total_chromosome_size = B.prod(B.tensor(self.x_matrix.shape[1:]))
    # flatten_shape = (self.x_matrix.shape[0],
    #                    int(total_chromosome_size))
    # self.flatten_x_matrix = B.reshape(self.x_matrix, flatten_shape)
    # B.shuffle(self.flatten_x_matrix)
    # B.shuffle(self.flatten_x_matrix.T)
    # self.x_matrix = B.reshape(self.flatten_x_matrix, self.x_matrix.shape)


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
    return cp.take(tensor, indices, axis=axis, out=out)


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
    idxs = cp.argpartition(tensor, k)[k:]

    # ! mind the - argsort return in increasing order we want largest
    return idxs[cp.argsort(-tensor[idxs])]


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
    idxs = cp.argpartition(tensor, k)[:k]
    return idxs[cp.argsort(tensor[idxs])]


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
