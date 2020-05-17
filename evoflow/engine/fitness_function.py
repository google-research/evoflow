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
import evoflow.backend as B
from collections import defaultdict


class FitnessFunction(object):
    """Fitness base classe.
    To be implemented by subclasses:
    * `call()`: Contains the logic for fitness calculation using `genes.
    """
    def __init__(self, **kwargs):
        self.debug = kwargs.get('debug', False)
        self._metrics = defaultdict(dict)

    def __call__(self, population):
        return self.call(population)

    @abc.abstractmethod
    def call(self, population):
        """Invokes the `Fitness` instance.

        Args:
            population: Tensor containing the population to assess
        """
        NotImplementedError('Must be implemented in subclasses.')

    def _flatten_population(self, population):
        """Convert the population to a 1D array as many ops (e.g norm) don't
        work on Ndimension
        """
        if len(population.shape) < 3:
            return population
        num_chromosomes = population.shape[0]
        flattened_size = int(B.prod(B.tensor(population.shape[1:])))
        return B.reshape(population, (num_chromosomes, flattened_size))

    def record_metric(self, name, value, group='default'):
        """Record metrics

        Args:
            name (str): name of the metrics.
            value (float): value to be recorded.
            group (str): group the metric belongs to. Used to group metrics
            together at display time.
        """
        self._metrics[group][name] = float(value)

    def get_metrics(self):
        """Return the lastest value of the recorded metrics
        """
        return self._metrics

    def print_debug(self, *msg):
        "output debug message"
        if self.debug:
            name = self.__class__.__name__
            print_debug(name, msg)
