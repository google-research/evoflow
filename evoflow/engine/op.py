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
from evoflow.utils import box, unbox
from evoflow.io import print_debug
from evoflow import backend as B
from evoflow.config import get_backend


class OP(object):
    "Base class for all operations"

    def __init__(self, **kwargs):

        # naming
        self.op_type = self.__class__.__name__
        self.idx = kwargs.get('name', self._gen_name())
        self.debug = kwargs.get('debug', False)
        # by default go as fast as possible
        self.OPTIMIZE = kwargs.get('optimize', True)

        # warm the user so there is no surprise
        if not self.OPTIMIZE:
            print('Optimizations disabled - execution will be slower')

        self.input_ops = []
        self.input_shapes = []  # track tensor size accross the ops.

        # optimization flags. They are set by each op based on what they can do
        # and what is the best way to call them. see dispatch()

        self.TF_FN = False  # by default don't use tf.functinon
        self.TF_XLA = False  # by default don't compile with XLA
        self.CPU_THRESHOLD = 0  # by default don't send small population to CPU

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
        results = self.dispatch(populations, self.GRAPH)
        return unbox(results)

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

        # FIXME: explore using a parallel map here
        results = []
        for population in populations:
            # FIXME: this is where the optimizaiton selection
            results.append(self.call(population))
        return results
        '''
                if self.use_tf:
            return self.compute_tf(population)
        else:
            return self.compute_cpu(population)

    def compute_cpu(self, population):
        return self._compute(population)

    @tf.function(experimental_compile=True)
    def compute_tf(self, population):
        return self._compute(population)
        '''

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

            # compute concrete results
            results = self.dispatch(ops, self.EAGER)

            # unbox result if needed
            return unbox(results)

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
