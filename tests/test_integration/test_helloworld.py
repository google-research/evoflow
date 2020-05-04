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

from geneflow import GeneFlow
import geneflow.backend as B
from geneflow.ops import RandomInputs, RandomMutations1D, UniformCrossover1D
from geneflow.selection import SelectFittest
from geneflow.fitness import SumGenes


def test_helloworld():
    "Solve the MAXONE problem"
    NUM_EVOLUTIONS = 10
    POPULATION_SIZE = 512
    GENE_SIZE = 1024

    inputs = RandomInputs((POPULATION_SIZE, GENE_SIZE), max_value=1)
    x = RandomMutations1D(max_gene_value=1)(inputs)
    outputs = UniformCrossover1D()(x)
    gf = GeneFlow(inputs, outputs, debug=0)
    fitness_function = SumGenes(expected_max_value=GENE_SIZE)
    evolution_strategy = SelectFittest()
    gf.compile(evolution_strategy, fitness_function)
    gf.summary()
    results = gf.evolve(num_evolutions=NUM_EVOLUTIONS)
    assert results

    metrics = results.get_metrics_history()

    # check metrics
    for metric_name, vals in metrics.items():
        assert isinstance(vals, list)
        assert len(vals) == 10
        assert isinstance(vals[9], float)

    assert 'fitness_mean' in metrics
    assert 'fitness_max' in metrics

    # assert the engine solved the (Trivial) problem
    assert max(metrics['fitness_max']) == 1
    assert metrics['fitness_max'][9] == 1
    assert min(metrics['fitness_max']) < 1  # max sure we did improve

    # check population value
    population = results.get_populations()
    assert (population.shape) == (POPULATION_SIZE, GENE_SIZE)
    assert B.tensor_equal(population[0], B.tensor([1] * GENE_SIZE))
