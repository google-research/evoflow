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
from termcolor import cprint
import tensorflow as tf

# ! MUST BE IN ITS OWN FILES TO BE TRACEABLE - DO NOT MOVE!
# https://github.com/tensorflow/tensorflow/issues/30409
# https://github.com/tensorflow/tensorflow/issues/34683


# only working version so far
def assign(dst_tensor, updates, indexes):
    """ assign values in tensor at the position specified by the slices.

    Args:
        dst_tensor (ndarray): Target tensor
        values (ndarray): Tensor containing the values to assign.
        indexes (list(list)): python array of the form:
        [[start][stop], ..., [start][stop]] where len == rank of dst_tensor.
        MUST be a python array not a tensor.

    Returns:
        ndarray: tensor with the assigned values.

    """
    # yes this code is verbose and can be simplified.. however its easier to
    # keep each case separated for debug and optim test idea purpose so leaving
    # it as in autograph simplifies it

    num_dims = len(indexes)
    updates = tf.reshape(updates, [-1])

    #  array size -> doesn't seems to provide any benefits
    arr_size = 1
    for d in indexes:
        arr_size *= (d[1] - d[0])

    v = 0
    indices = tf.TensorArray(dtype=tf.int32,
                             size=arr_size,
                             dynamic_size=False,
                             element_shape=(num_dims, ))

    if num_dims == 2:
        for d1 in range(indexes[0][0], indexes[0][1]):
            for d2 in range(indexes[1][0], indexes[1][1]):
                indices = indices.write(v, [d1, d2])
                v += 1

    elif num_dims == 3:
        for d1 in tf.range(indexes[0][0], indexes[0][1]):
            for d2 in tf.range(indexes[1][0], indexes[1][1]):
                for d3 in tf.range(indexes[2][0], indexes[2][1]):
                    indices = indices.write(v, [d1, d2, d3])
                    v += 1

    elif num_dims == 4:
        for d1 in tf.range(indexes[0][0], indexes[0][1]):
            for d2 in tf.range(indexes[1][0], indexes[1][1]):
                for d3 in tf.range(indexes[2][0], indexes[2][1]):
                    for d4 in tf.range(indexes[3][0], indexes[3][1]):
                        indices = indices.write(v, [d1, d2, d3, d4])
                        v += 1
    else:
        raise ValueError('Not implemented for rank >2')

    indices = indices.stack()  # to tensor
    return tf.tensor_scatter_nd_update(dst_tensor,
                                       indices=indices,
                                       updates=updates)


# ! not working tensor placement error again
def assign_batched_failed(dst_tensor, updates, indexes):
    """ assign values in tensor at the position specified by the slices.

    Args:
        dst_tensor (ndarray): Target tensor
        values (ndarray): Tensor containing the values to assign.
        indexes (list(list)): python array of the form:
        [[start][stop], ..., [start][stop]] where len == rank of dst_tensor.
        MUST be a python array not a tensor.

    Returns:
        ndarray: tensor with the assigned values.

    """
    # yes this code is verbose and can be simplified.. however its easier to
    # keep each case separated for debug and testing optim ideas purpose
    # it as in autograph simplifies it

    num_dims = len(indexes)
    updates = tf.reshape(updates, [-1])

    #  array size
    arr_size = 1
    for d in indexes[:-1]:
        arr_size *= (d[1] - d[0])

    v = 0
    indices = tf.TensorArray(dtype=tf.int32, size=arr_size, dynamic_size=False)
    # ,element_shape=(num_dims * last_dim_size, ))

    if num_dims == 3:
        for d1 in tf.range(indexes[0][0], indexes[0][1]):
            for d2 in tf.range(indexes[1][0], indexes[1][1]):

                # try to minimize the update by batching it
                update = []
                for d3 in range(indexes[2][0], indexes[2][1]):
                    update.append([d1, d2, d3])

                indices = indices.write(v, update)
                v += 1
    else:
        raise ValueError('Not implemented for rank 1 and >4')

    indices = indices.stack()  # to tensor
    cprint(indices, 'yellow')
    cprint(indices.shape, 'blue')
    return tf.tensor_scatter_nd_update(dst_tensor,
                                       indices=indices,
                                       updates=updates)


# slower
def assign2(dst_tensor, updates, indexes):
    """ assign values in tensor at the position specified by the slices.

    Args:
        dst_tensor (ndarray): Target tensor
        values (ndarray): Tensor containing the values to assign.
        indexes (list(list)): python array of the form:
        [[start][stop], ..., [start][stop]] where len == rank of dst_tensor.
        MUST be a python array not a tensor.

    Returns:
        ndarray: tensor with the assigned values.

    """
    # yes this code is verbose and can be simplified.. however its easier to
    # keep each case separated for debug and optim test idea purpose so leaving
    # it as in autograph simplifies it

    num_dims = len(indexes)
    updates = tf.reshape(updates, [-1])

    # passing arr_size is also not working
    # Not working
    #  array size
    # arr_size = 1
    # for d in indexes:
    #     arr_size *= (d[1] - d[0])
    # print(arr_size)
    if num_dims == 2:
        indices = tf.TensorArray(dtype=tf.int32,
                                 size=0,
                                 dynamic_size=True,
                                 element_shape=(1, ))
        for d1idx in range(indexes[0][0], indexes[0][1]):
            for d2idx in range(indexes[1][0], indexes[1][1]):
                indices = indices.write(indices.size(), d1idx)
                indices = indices.write(indices.size(), d2idx)

    elif num_dims == 3:
        indices = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        for d1idx in tf.range(indexes[0][0], indexes[0][1]):
            for d2idx in tf.range(indexes[1][0], indexes[1][1]):
                for d3idx in tf.range(indexes[2][0], indexes[2][1]):
                    indices = indices.write(indices.size(), d1idx)
                    indices = indices.write(indices.size(), d2idx)
                    indices = indices.write(indices.size(), d3idx)
    else:
        raise ValueError('Not implemented for rank >2')

    arr_len = indices.size()
    indices = indices.stack()
    dim_size = tf.cast(arr_len / num_dims, 'int32')
    indices = tf.reshape(indices, (dim_size, num_dims))
    # cprint(arr_len, 'blue')
    # cprint(dim_size, 'yellow')
    # cprint(indices.shape, 'green')

    return tf.tensor_scatter_nd_update(dst_tensor,
                                       indices=indices,
                                       updates=updates)


# not working with randint but wish it would
def assign_broken(dst_tensor, updates, indexes):
    """ assign values in tensor at the position specified by the slices.

    Args:
        dst_tensor (ndarray): Target tensor
        values (ndarray): Tensor containing the values to assign.
        indexes (list(list)): python array of the form:
        [[start][stop], ..., [start][stop]] where len == rank of dst_tensor.
        MUST be a python array not a tensor.

    Returns:
        ndarray: tensor with the assigned values.

    """
    # !gotcha:
    # ! - slices must be a python array
    # ! - do not use tf.range
    # ! - need its own file (why???)

    # yes this code is verbose and can be simplified.. however its easier to
    # keep each case separated for debug and optim test idea purpose so leaving
    # it as in autograph simplifies it

    num_dims = len(indexes)
    updates = tf.reshape(updates, [-1])

    indices = []
    if num_dims == 1:
        for idx in range(indexes[0][0], indexes[0][1]):
            indices.append(idx)

    elif num_dims == 2:
        for d1idx in range(indexes[0][0], indexes[0][1]):
            for d2idx in range(indexes[1][0], indexes[1][1]):
                indices.append([d1idx, d2idx])
    elif num_dims == 3:
        for d1idx in range(indexes[0][0], indexes[0][1]):
            for d2idx in range(indexes[1][0], indexes[1][1]):
                for d3idx in range(indexes[2][0], indexes[2][1]):
                    indices.append([d1idx, d2idx, d3idx])

    elif num_dims == 4:
        for d1idx in range(indexes[0][0], indexes[0][1]):
            for d2idx in range(indexes[1][0], indexes[1][1]):
                for d3idx in range(indexes[2][0], indexes[2][1]):
                    for d4idx in range(indexes[3][0], indexes[3][1]):
                        indices.append([d1idx, d2idx, d3idx, d4idx])
    else:
        raise ValueError('Not implemented for rank >2')

    # cprint(indexes, 'blue')
    # cprint(indices, 'yellow')
    # cprint(updates.shape, 'green')

    return tf.tensor_scatter_nd_update(dst_tensor,
                                       indices=indices,
                                       updates=updates)