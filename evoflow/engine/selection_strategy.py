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

import abc
from evoflow.io import print_debug


class SelectionStrategy(object):
    """Selection Strategy base classe.
    To be implemented by subclasses:
    * `call()`: Contains the logic for selecting which chromesome to keep.
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.debug = kwargs.get('debug', False)

    @abc.abstractmethod
    def call(self, fitness_function, current_population, evolved_population):
        """Select new population.

        Args:
            fitness_function (function): User provided function that return
            the fitness value for each chromosome of a population as a Tensor.

            current_population: Tensor containing the population prior to
            evolution.

            evolved_population: Tensor containing the population after the
            evolution

        """
        NotImplementedError('Must be implemented in subclasses.')

    def __call__(self, fitness_function, current_population,
                 evolved_population):
        """Select new population.

        Args:
            fitness_function (function): User provided function that return
            the fitness value for each chromosome of a population as a Tensor.

            current_population: Tensor containing the population prior to
            evolution.

            evolved_population: Tensor containing the population after the
            evolution

        Returns
            [new_populations, fitness_values]

        """
        return self.call(fitness_function, current_population,
                         evolved_population)

    def print_debug(self, *msg):
        "output debug message"
        if self.debug:
            name = self.__class__.__name__
            print_debug(name, msg)
