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

import evoflow.backend as B
from evoflow.engine import FitnessFunction


class Sum(FitnessFunction):
    def __init__(self, max_sum_value=None, **kwargs):
        """Compute the sum of the chromosomes as fitness values.

        Args:
            expected_max_sum_value (int, optional): What is the maximum value
            the sum per chromosome will reach. Used to normalize the fitness
            function between 0 and 1 if specified. Defaults to None.

        Note:
            This fitness function is used to solve the MAXONE problem.

        """
        super(Sum, self).__init__(**kwargs)
        if max_sum_value:
            self.max_sum_value = B.tensor(max_sum_value)
        else:
            self.max_sum_value = None

    def call(self, population):
        flat_pop = self._flatten_population(population)
        scores = B.sum(flat_pop, axis=-1)
        if self.max_sum_value:
            return scores / self.max_sum_value
        else:
            return scores
