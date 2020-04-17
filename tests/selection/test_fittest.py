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
from geneflow.selection import SelectFittest
from geneflow.fitness import CosineSimilarity


def test_fittest():

    r = B.tensor([2, 0, 1, 1, 0, 2, 1, 1])
    d = B.tensor([2, 1, 1, 0, 1, 1, 1, 1])
    pop = B.tensor([r, d, r, d, r, d])

    fitness_function = CosineSimilarity(r)
    selector = SelectFittest()
    selected_pop, fitness_scores = selector(fitness_function, pop, pop)
    print(selected_pop)
    assert selected_pop.shape == pop.shape
    for chromosome in selected_pop:
        assert B.tensor_equal(chromosome, r)
