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

from geneflow.engine import OP
from geneflow import backend as B


class Inputs(OP):

    def __init__(self, shape, **kwargs):
        """Use a supplied set genomes

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
            raise ValueError('Incompatible input shape expected: %s - got: %s'
                             % (self.shape, chromosomes.shape))
        self.chromosomes = chromosomes

    def call(self):
        "Return input values"
        return self.chromosomes


class RandomInputs(Inputs):

    def __init__(self, shape, min_value=0, max_value=256,
                 always_regnerate=False, **kwargs):
        """Generate genomes as random integer inputs

        Args:
            shape ([type]): [description]
            min_value (int, optional): [description]. Defaults to 0.
            max_value (int, optional): [description]. Defaults to 256.
            always_regenerate(bool optional): Always generate a new set of
            random inputs when calling compute(). Defaults to False.
        """
        super(RandomInputs, self).__init__(shape, **kwargs)

        self.min_value = min_value
        self.max_value = max_value + 1
        self.always_regenerate = always_regnerate
        self.chromosomes = self._generate_chromosomes()

    def call(self, regenerate=False):
        """Provide random inputs based on the specified shape

        Args:
            regenerate (bool, optional): Draw a new set of random inputs.
            Defaults to False.

        Returns:
            ndarray: Tensor containing random chromosomes or assigned ones.

        Note:
            Never regenerates chromosome between call by default. this will
            breaks the algorithm which relies on the inputs tensor returning
            the latest values

        """
        if regenerate or self.always_regenerate:
            self.chromosomes = self._generate_chromosomes()
        return self.chromosomes

    def _generate_chromosomes(self):
        "Generate a random genepool"
        # FIXME support other type of input (float)
        return B.randint(self.min_value, self.max_value, shape=self.shape)
