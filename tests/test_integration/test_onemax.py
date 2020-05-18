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

from evoflow.engine import EvoFlow
import evoflow.backend as B
from evoflow.ops import Input, RandomMutations1D, UniformCrossover1D
from evoflow.selection import SelectFittest
from evoflow.fitness import Sum
from evoflow.population import randint_population
from evoflow.callbacks import DummyCallback


def test_helloworld():
    "Solve the MAXONE problem"
    NUM_EVOLUTIONS = 10
    SHAPE = (512, 1024)
    MAX_VAL = 1

    population = randint_population(SHAPE, MAX_VAL)

    inputs = Input(SHAPE)
    x = RandomMutations1D(max_gene_value=1)(inputs)
    outputs = UniformCrossover1D()(x)
    gf = EvoFlow(inputs, outputs, debug=0)
    fitness_function = Sum(max_sum_value=SHAPE[1])
    evolution_strategy = SelectFittest()
    gf.compile(evolution_strategy, fitness_function)
    gf.summary()
    results = gf.evolve(population,
                        generations=NUM_EVOLUTIONS,
                        callbacks=[DummyCallback()])
    assert results

    metrics = results.get_metrics_history()

    # check metrics
    for metric_grp, data in metrics.items():

        for metric_name, vals in data.items():
            assert isinstance(vals, list)
            if metric_grp == 'Fitness function':
                assert len(vals) == 10
                assert isinstance(vals[9], float)

                assert 'mean' in data
                assert 'max' in data

    # assert the engine solved the (Trivial) problem
    max_fitness = metrics['Fitness function']['max']
    assert max(max_fitness) == 1
    assert max_fitness[9] == 1
    assert min(max_fitness) < 1  # max sure we did improve

    # check population value
    population = results.get_populations()
    assert (population.shape) == SHAPE
    assert B.tensor_equal(population[0], B.tensor([1] * SHAPE[1]))
