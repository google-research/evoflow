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

import numpy as np
import geneflow.backend as B
from geneflow.selection import SelectFittest
from geneflow.fitness import InvertedCosineSimilarity


def test_fittest():

    ref = B.tensor([2, 0, 1, 1, 0, 2, 1, 1])
    d = B.tensor([2, 1, 1, 0, 1, 1, 1, 1])
    pop = B.tensor([ref, d, ref, d, ref, d])

    fitness_function = InvertedCosineSimilarity(ref)
    selector = SelectFittest()
    selected_pop, fitness_scores, metrics = selector(fitness_function, pop,
                                                     pop)
    print(selected_pop)
    assert selected_pop.shape == pop.shape
    for chromosome in selected_pop:
        assert B.tensor_equal(chromosome, ref)


def test_fittest_2d():

    INSERTION_POINTS = [0, 10, 20]  # where we copy the ref chromosome

    # using numpy as fancy indexing in TF is a pain and perf is not critical.
    ref_chromosome = np.random.randint(0, 2, (32, 32))
    pop1 = np.random.randint(0, 2, (64, 32, 32))
    pop2 = np.random.randint(0, 2, (64, 32, 32))
    for idx in INSERTION_POINTS:
        pop1[idx] = ref_chromosome

    ref_chromosome = B.tensor(ref_chromosome)
    pop1 = B.tensor(pop1)
    pop2 = B.tensor(pop2)

    fitness_function = InvertedCosineSimilarity(ref_chromosome)
    selector = SelectFittest()
    selected_pop, fitness_scores, metrics = selector(fitness_function, pop1,
                                                     pop2)
    print(selected_pop)
    assert selected_pop.shape == pop1.shape
    # check the exact chromosome is the three top choice
    assert B.tensor_equal(selected_pop[0], ref_chromosome)
    assert B.tensor_equal(selected_pop[1], ref_chromosome)
    assert B.tensor_equal(selected_pop[2], ref_chromosome)
