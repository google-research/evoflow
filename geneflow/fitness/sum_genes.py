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

import geneflow.backend as B
from geneflow.engine import FitnessFunction


class SumGenes(FitnessFunction):

    def __init__(self, expected_max_value=None, **kwargs):
        """Compute the sum of the gene as fitness value.

        Args:
            expected_max_value (int): expected max value if known. Used
            to normalize the fitness function if specified.

        """
        super(SumGenes, self).__init__('sum_genes', **kwargs)
        self.expected_max_value = B.tensor(expected_max_value)

    def call(self, population):
        if self.expected_max_value:
            return B.sum(population, axis=-1) / self.expected_max_value
        else:
            return B.sum(population, axis=-1)
