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

from geneflow.ops import RandomMutations
from geneflow import backend as B
from copy import copy


def test_mutation_graph_mode():
    "make sure the boxing / unboxing works in graph mode"
    W = 8
    GENOME_SHAPE = (W, W)

    min_mutation_value = -3
    max_mutation_value = 3

    chromosomes = B.randint(0, 100, GENOME_SHAPE)

    RM = RandomMutations(max_gene_value=100,
                         min_mutation_value=min_mutation_value,
                         max_mutation_value=max_mutation_value)
    chromosomes = RM._call_from_graph(chromosomes)
    assert B.is_tensor(chromosomes)
    assert chromosomes.shape == GENOME_SHAPE


def test_mutation_zero_one():
    W = 8
    GENOME_SHAPE = (W, W)

    chromosomes = B.randint(0, 2, GENOME_SHAPE)
    original = copy(chromosomes)
    RM = RandomMutations(max_gene_value=1, debug=0)

    for _ in range(10):
        chromosomes = RM(chromosomes)
        diff = original - chromosomes
        assert B.sum(B.abs(diff)) == W
        original = copy(chromosomes)


def test_mutation():
    W = 8
    GENOME_SHAPE = (W, W)
    MAX_GENE_VAL = 100
    min_mutation_value = -3
    max_mutation_value = 3

    chromosomes = B.randint(0, MAX_GENE_VAL, GENOME_SHAPE)

    RM = RandomMutations(min_mutation_value=min_mutation_value,
                         max_mutation_value=max_mutation_value)
    chromosomes = RM(chromosomes)
    assert B.is_tensor(chromosomes)
    assert chromosomes.shape == GENOME_SHAPE
    chromosomes_sav = copy(chromosomes)
    chromosomes = RM(chromosomes)
    diff = chromosomes - chromosomes_sav

    assert B.max(diff) <= max_mutation_value
    assert B.min(diff) >= min_mutation_value
