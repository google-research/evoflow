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

import tensorflow as tf
from perfcounters import PerfCounters


@tf.function()
def randint_tf(low, high, shape):
    return B.randint(low, high, shape)


@tf.function(experimental_compile=True)
def randint_xla(low, high, shape):
    return B.randint(low, high, shape)


def test_randint_fn():
    NUM_TESTS = 3
    low = 10
    high = 1000
    shape = (100, 100, 100)

    randint_tf(low, high, shape)
    randint_xla(low, high, shape)

    v = randint_tf(low, high, shape)
    v2 = randint_tf(low, high, shape)
    assert not B.tensor_equal(v, v2)

    v = randint_xla(low, high, shape)
    v2 = randint_xla(low, high, shape)
    assert not B.tensor_equal(v, v2)

    cnts = PerfCounters()
    cnts.start('basic')
    for _ in range(NUM_TESTS):
        B.randint(low, high, shape)
    cnts.stop('basic')

    cnts.start('tf_fn')
    for _ in range(NUM_TESTS):
        randint_tf(low, high, shape)
    cnts.stop('tf_fn')

    cnts.start('tf_xla')
    for _ in range(NUM_TESTS):
        randint_xla(low, high, shape)
    cnts.stop('tf_xla')

    cnts.report()
