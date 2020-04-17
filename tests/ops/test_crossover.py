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

# import os
# os.environ['GENEFLOW_BACKEND'] = 'numpy'

from geneflow import backend as B  # noqa: F402
from geneflow.ops import UniformCrossover
from geneflow.ops import DualCrossover
from geneflow.ops import SingleCrossover


def test_uniform_crossover():
    GENOME_SHAPE = (64, 128)
    chromosomes = B.randint(0, 1024, GENOME_SHAPE)
    num_crossover_fraction = 0.5
    crossover_size_fraction = 0.2

    # peforming crossover
    chromosomes, ref_chromosomes = UniformCrossover(num_crossover_fraction,
                                                    crossover_size_fraction,
                                                    debug=True)(chromosomes)

    # measuring mutation rate
    diff = B.clip(B.abs(chromosomes - ref_chromosomes), 0, 1)
    # cprint(diff, 'cyan')

    bins = B.bincount(diff.flatten())
    mutation_rate = int(bins[1] / sum(bins) * 100)

    # computing expected mutation rate
    expected_mutation_rate = num_crossover_fraction
    expected_mutation_rate *= crossover_size_fraction
    expected_mutation_rate *= 100

    # diff
    mutation_rate_diff = abs(mutation_rate - expected_mutation_rate)

    assert mutation_rate_diff < 2
    assert chromosomes.shape == GENOME_SHAPE


def test_crossover_graph_mode():
    "make sure the boxing / unboxing works in graph mode"
    GENOME_SHAPE = (64, 128)
    chromosomes = B.randint(0, 1024, GENOME_SHAPE)
    num_crossover_fraction = 0.5
    crossover_size_fraction = 0.2
    print('pre_call', chromosomes.shape)
    SC = SingleCrossover(num_crossover_fraction, crossover_size_fraction)

    chromosomes = SC._call_from_graph(chromosomes)
    assert chromosomes.shape == GENOME_SHAPE


def test_single_crossover_eager():
    GENOME_SHAPE = (64, 128)
    chromosomes = B.randint(0, 1024, GENOME_SHAPE)
    num_crossover_fraction = 0.5
    crossover_size_fraction = 0.2

    chromosomes, ref_chromosomes = SingleCrossover(num_crossover_fraction,
                                                   crossover_size_fraction,
                                                   debug=1)(chromosomes)

    assert chromosomes.shape == GENOME_SHAPE
    # measuring mutation rate
    diff = B.clip(B.abs(chromosomes - ref_chromosomes), 0, 1)
    # cprint(diff, 'cyan')

    # row test
    num_ones_in_row = 0
    for col in diff:
        num_ones_in_row = max(list(col).count(1), num_ones_in_row)

    max_one_in_row = GENOME_SHAPE[1] * crossover_size_fraction
    assert num_ones_in_row <= max_one_in_row
    assert num_ones_in_row

    # col
    diff = diff.T
    num_ones_in_col = 0
    for col in diff:
        num_ones_in_col = max(list(col).count(1), num_ones_in_col)

    max_one_in_col = GENOME_SHAPE[0] * num_crossover_fraction
    assert max_one_in_col - 2 <= num_ones_in_col <= max_one_in_col


def test_dual_crossover():
    GENOME_SHAPE = (64, 128)
    chromosomes = B.randint(0, 1024, GENOME_SHAPE)
    num_crossover_fraction = 0.5
    crossover_size_fraction = 0.2

    chromosomes, ref_chromosomes = DualCrossover(num_crossover_fraction,
                                                 crossover_size_fraction,
                                                 debug=True)(chromosomes)

    # measuring mutation rate
    diff = B.clip(abs(chromosomes - ref_chromosomes), 0, 1)
    # cprint(diff, 'cyan')

    # col
    diff = diff.T
    num_ones_in_col = 0
    for col in diff:
        num_ones_in_col = max(list(col).count(1), num_ones_in_col)

    max_one_in_col = GENOME_SHAPE[0] * num_crossover_fraction
    assert max_one_in_col - 2 <= num_ones_in_col <= max_one_in_col
