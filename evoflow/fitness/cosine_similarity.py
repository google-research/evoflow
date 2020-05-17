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


class InvertedCosineSimilarity(FitnessFunction):
    def __init__(self, reference_chromosome, **kwargs):
        """Inverted Cosine similarity function that returns 1 when chromosomes
        are similar to the reference chromose and [0, 1[ when different

        For reference implementation see
        https://github.com/scipy/scipy/blob/v0.14.0/scipy/spatial/distance.py#L267  # noqa

        Args:
            reference_chromosome (tensor1D): reference_chromosome.
        """
        super(InvertedCosineSimilarity, self).__init__(**kwargs)

        # cache ref chromosome flattend
        self.ref_chromosome = B.flatten(B.tensor(reference_chromosome))

        # caching ref pop norm
        self.ref_norm = B.norm(B.cast(self.ref_chromosome, B.floatx()))

    def call(self, population):
        # cosine only works on 1D array we need to flatten
        flat_pop = self._flatten_population(population)

        # numerator
        numerator = B.cast(B.dot(flat_pop, self.ref_chromosome), B.floatx())
        # denominator (norm(u) *  norm(v))
        # ! B.norm is not broadcasted we need our own version
        population_norm = B.broadcasted_norm(flat_pop)

        # population_norm = B.norm(B.cast(flat_pop, B.floatx()), axis=0)
        denominator = population_norm * self.ref_norm

        return (numerator / denominator)
