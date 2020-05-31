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

from evoflow.io import print_debug


class Callback(object):
    """Abstract base class used to build new callbacks.
 """
    def __init__(self):
        self.model = None

    def on_evolution_begin(self, populations):
        """Called at the start of the evolution

        Args:
            populations (Tensors): Original population
        """
        pass

    def on_evolution_end(self, populations):
        """Called at the end of the evolution

        Args:
            populations (Tensors): final population
        """
        pass

    def on_generation_begin(self, generation):
        """Called at the start of a generation.

        Args:
            generation_idx (int): Generation.
        """
        pass

    def on_generation_end(self, generation, metrics, fitness_scores,
                          populations):
        """Called at the end of each generation.

        Args:
            generation (int): Generation index
            metrics ([type]): dictionnary containing user defined metrics
            fitness_scores (dict): dictionnary contraining the fitness scores
            populations (list): evolved populations
        """
        pass

    def set_model(self, model):
        "Assign current evoflow model"
        self.model = model

    def print_debug(self, *msg):
        "output debug message"
        if self.debug:
            name = self.__class__.__name__
            print_debug(name, msg)
