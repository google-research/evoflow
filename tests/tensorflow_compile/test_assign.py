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

import evoflow.backend.tensorflow as B

# from evoflow.backend.tf_ops.assign import assign as assign2
from evoflow.backend.common import intx
from evoflow.utils import slices2array
import tensorflow as tf
import numpy as np
from perfcounters import PerfCounters

# @tf.function()
# def assign_tf(dst_tensor, updates, indexes):
#     return B.assign(dst_tensor, updates, indexes)


@tf.function()
def assign_tf(dst_tensor, values, slices):
    return B.assign(dst_tensor, values, slices)


@tf.function(experimental_compile=True)
def assign_xla(dst_tensor, values, slices):
    return B.assign(dst_tensor, values, slices)


def test_assign_fn():
    NUM_TESTS = 3
    t = np.random.randint(0, 100, (100, 100))
    dst_tensor = np.zeros(t.shape, dtype=intx())
    slices = (slice(1, 30), slice(1, 25))
    indexes = slices2array(slices)
    updates = B.cast(t[slices], dtype=intx())

    r = B.assign(dst_tensor, updates, indexes)
    r2 = assign_tf(dst_tensor, updates, indexes)
    assert B.tensor_equal(r, r2)

    assign_tf(dst_tensor, updates, indexes)
    assign_xla(dst_tensor, updates, indexes)

    cnts = PerfCounters()
    cnts.start('basic')
    for _ in range(NUM_TESTS):
        B.assign(dst_tensor, updates, indexes)
    cnts.stop('basic')

    cnts.start('tf_fn')
    for _ in range(NUM_TESTS):
        assign_tf(dst_tensor, updates, indexes)
    cnts.stop('tf_fn')

    cnts.start('tf_xla')
    for _ in range(NUM_TESTS):
        assign_xla(dst_tensor, updates, indexes)
    cnts.stop('tf_xla')

    cnts.report()
