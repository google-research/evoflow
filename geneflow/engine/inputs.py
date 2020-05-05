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

from .op import OP


class Inputs(OP):
    def __init__(self, shape, **kwargs):
        """Base class for input
        #! do not use directly

        Args:
            shape (set of ints): what is the shape of the input. Usually it is
            (num_chromosomes, num_genes)

        Note:
            Never regenerates chromosome between call by default. this will
            breaks the algorithm which relies on the inputs tensor returning
            the latest values

        """
        self.shape = shape
        super(Inputs, self).__init__(**kwargs)

    def get_output_shapes(self):
        "return the shape of tensor returned by the op"
        return self.shape

    def assign(self, chromosomes):
        """Assign concrete values to the input
        """
        if chromosomes.shape != self.shape:
            raise ValueError(
                'Incompatible input shape expected: %s - got: %s' %
                (self.shape, chromosomes.shape))
        self.chromosomes = chromosomes

    def call(self):
        "Return input values"
        return self.chromosomes
