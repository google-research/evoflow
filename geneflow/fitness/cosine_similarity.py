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


class InvertedCosineSimilarity(FitnessFunction):
    def __init__(self, reference_chromosome, **kwargs):
        """Inverted Cosine similarity function that returns 1 when chromosomes
        are similar to the reference chromose and [0, 1[ when different

        For reference implementation see
        https://github.com/scipy/scipy/blob/v0.14.0/scipy/spatial/distance.py#L267  # noqa

        Args:
            reference_chromosome (tensor1D): reference_chromosome.
        """
        super(InvertedCosineSimilarity, self).__init__('invet_cosine_sim',
                                                       **kwargs)

        # cache ref chromosome flattend
        self.ref_chromosome = B.flatten(B.tensor(reference_chromosome))

        # caching ref pop norm
        self.ref_norm = B.norm(self.ref_chromosome)

    def call(self, population):
        # cosine only works on 1D array we need to flatten
        flat_pop = self._flatten_population(population)

        print('flat_pop', flat_pop.shape)
        print('ref_chromosome', self.ref_chromosome.shape)
        # numerator
        numerator = B.dot(flat_pop, self.ref_chromosome)

        # denominator (norm(u) *  norm(v))
        # ! B.norm is not broadcasted we need our own version
        population_norm = B.sum(B.abs(flat_pop)**2, axis=-1)**0.5
        denominator = population_norm * self.ref_norm

        print(numerator)
        print(denominator)
        return (numerator / denominator)
