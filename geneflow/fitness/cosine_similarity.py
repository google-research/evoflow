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


class CosineSimilarity(FitnessFunction):

    def __init__(self, reference_chromosome, **kwargs):
        """Compute how similar a population is to a reference chromosome.

        Args:
            reference_chromosome (tensor): reference_chromosome.

        # FIXME: normalize value betwen 0/1 by using the reference chromosome
        """
        super(CosineSimilarity, self).__init__('cosine_similarity', **kwargs)

        self.reference_chromosome = B.tensor(reference_chromosome)

        # caching the norm computation
        self.reference_chromosome_norm = B.norm(self.reference_chromosome)

    def call(self, population):
        numerator = B.dot(population, self.reference_chromosome)
        # ! B.norm is not broadcasted we need our own version
        population_norm = B.sum(B.abs(population) ** 2, axis=-1) ** 0.5

        denominator = population_norm * self.reference_chromosome_norm
        return numerator / denominator
