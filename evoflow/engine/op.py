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

import abc
import random
import hashlib
from time import time
from evoflow.utils import box, unbox, optimization_support
from evoflow.io import print_debug
from evoflow import backend as B
from evoflow.config import get_backend

import tensorflow as tf


class OP(object):
    "Base class for all operations"

    def __init__(self, **kwargs):

        # naming
        self.op_type = self.__class__.__name__
        self.idx = kwargs.get('name', self._gen_name())
        self.debug = kwargs.get('debug', False)
        # by default go as fast as possible

        self.input_ops = []
        self.input_shapes = []  # track tensor size accross the ops.

        # optimization flags. They are set by each op based on what they can do
        # and what is the best way to call them. see dispatch()

        self.CPU_THRESHOLD = 0  # by default don't send small population to CPU

        # max optimization level
        self.OPTIMIZE_LEVEL = 2  # default to max
        self.set_optimization_level(kwargs.get('optimization_level', 2))

        # infered env flags needed to make dispatch decisions
        self.TF = False  # by default we don't use any TF specific optimization
        self.TF_GPU = 0  # by default don't have gpu

        # check if we use TF and we use GPU
        if get_backend() == 'tensorflow':
            self.TF = True  # using TF so can enable specific optimization
            from evoflow.backend.tensorflow import get_num_gpu
            self.TF_GPU = get_num_gpu()  # using GPU so can enable GPU optim

        # track execution type
        self.EAGER = 1
        self.GRAPH = 2

    def set_optimization_level(self, level):
        "Change optimization level"
        self.OPTIMIZE_LEVEL = level

        # warm the user so there is no surprise
        if not self.OPTIMIZE_LEVEL:
            print('Optimization are disabled - execution will be slower')
        elif self.OPTIMIZE_LEVEL == 1:
            print('Some optimization are disabled - execution will be slower')

    @abc.abstractmethod
    def call(self, population, **kwargs):
        """This is where the logic of the operation live.

        Args:
            population (ndarrays): Tensor
            **kwargs: Additional keyword arguments to be passed to `call()`.
        """
        return population

    def _gen_name(self):
        return self.op_type.lower() + "_" + self._gen_idx()

    def _call_from_graph(self, populations):
        "Function called during graph executions"
        populations = box(populations)
        # dispatch take care of iterating through populations
        return unbox(self.dispatch(populations, self.GRAPH))

    def dispatch(self, populations, mode):
        """Fan-out populations and call the most efficient compute method based
        on what optimization the op support

        Note: it is each OP responsability to set optimization flags at init
        time so dispatch know if tf.function, xla etc should be used.

        Args:
            populations (list(tensors)): populations to apply the op to.
            mode (int): Mode of operation in {EAGER, GRAPH}
        Returns:
            list(tensors): mutated populations
        """

        optim_support = optimization_support(self)
        AUTOGRAPH = optim_support['optimizations']['autograph']
        XLA = optim_support['optimizations']['xla']

        self.print_debug('TF', self.TF, 'TF_GPU', self.TF_GPU, 'AUTOGRAPH',
                         AUTOGRAPH, 'XLA', XLA, 'Requested level',
                         self.OPTIMIZE_LEVEL)

        if not self.OPTIMIZE_LEVEL or not self.TF:
            self.print_debug('Optimization disabled or not TF')
            fn = self.call
        else:
            if XLA and self.OPTIMIZE_LEVEL >= 2:
                self.print_debug('XLA optimizations')
                fn = self.tf_xla_call
            elif AUTOGRAPH:
                self.print_debug('AUTOGRAPH optimizations')
                fn = self.tf_call
            else:
                self.print_debug('No optimization exist for this op')
                fn = self.call

            # FIXME: add GPU versus CPU placement here (inside the else)

        # FIXME: explore using a parallel map here
        results = []
        for population in populations:
            results.append(fn(population))
        return results

    @tf.function()
    def tf_call(self, population):
        return self.call(population)

    @tf.function(experimental_compile=True)
    def tf_xla_call(self, population):
        return self.call(population)

    def __call__(self, ops):

        # boxing inputs if needed
        ops = box(ops)

        # graph computation or eager?
        if self.debug:
            input_types = [type(op) for op in ops]
            self.print_debug('inputs type:%s' % (input_types))

        if issubclass(type(ops[0]), OP):
            # graph mode
            self.print_debug('graph mode')

            input_shapes = []
            for op in ops:
                assert issubclass(type(op), OP)
                self.input_ops.append(op)
                input_shapes.append(op.get_output_shapes())

            self.compute_output_shape(input_shapes)

            return self
        else:
            # eager mode
            self.print_debug('eager mode')

            # check inputs are valis
            for op in ops:
                if not B.is_tensor(op):
                    raise ValueError("Expecting list(tensors) or a tensor")

            # input shapes
            input_shapes = []
            for op in ops:
                input_shapes.append(op.shape)
            self.compute_output_shape(input_shapes)

            # compute concrete results - dispatch iterate through populations.
            return unbox(self.dispatch(ops, self.EAGER))

    def compute_output_shape(self, input_shapes):
        """Compute output shapes

        Args:
            list(tuples): input shapes

        Note:
            any kind of initialization depending of input shape can be done by
            redefining this function in the children class.
        """

        self.output_shapes = input_shapes

    def get_output_shapes(self):
        "return the shape of tensor returned by the op"
        return unbox(self.input_shapes)

    def _gen_idx(self):
        "generate a unique idx"
        idx = "%s%s" % (int(time()), random.randint(10000000, 90000000))
        idx = hashlib.md5(idx.encode()).hexdigest()[:6].upper()
        return idx

    def get_config(self):
        "export op config"
        config = {
            "node_type": self.op_type,
            "idx": self.idx,
            "inbound_ops": self.inbound_ops
        }
        return config

    def print_debug(self, *msg):
        "output debug message"
        if self.debug:
            print_debug(self.idx, msg)

    @classmethod
    def from_config(cls, config):
        "create an op from its config"
        return cls(**config)
