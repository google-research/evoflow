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
from evoflow.engine import SelectionStrategy


class SelectFittest(SelectionStrategy):
    "Select the fittest member of the population"

    def __init__(self, mode='max', **kwargs):
        """[summary]

        Args:
            mode (str, optional):  one of `{min', 'max'}`. In 'min' mode,
            the fitness function will select individual with the lowest fitness
            value; in 'max' mode it will select the one with the highest
            values. Defaults to 'max'.
        """
        if mode not in ['min', 'max']:
            raise ValueError('mode must be either max or min')

        self.mode = mode
        super(SelectFittest, self).__init__('select_fittest', **kwargs)

    def call(self, fitness_function, current_population, evolved_population):
        """Select the most fit individuals from the combined current and
        evolved population.

        Args:
            fitness_function (function): User provided function that return
            the fitness value for each chromosome of a population as a Tensor.

            current_population: Tensor containing the population prior to
            evolution.

            evolved_population: Tensor containing the population after the
            evolution
        """
        population_size = current_population.shape[0]
        merged_population = B.concatenate(
            [current_population, evolved_population])

        # fitness computation
        fitness_scores = fitness_function(merged_population)
        metrics = fitness_function.get_metrics()

        # population selection
        if self.mode == 'max':
            indices = B.top_k_indices(fitness_scores, k=population_size)
        else:
            indices = B.bottom_k_indices(fitness_scores, k=population_size)

        selected_pop = B.take(merged_population, indices, axis=0)

        return selected_pop, B.take(fitness_scores, indices, axis=0), metrics
