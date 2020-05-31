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

from evoflow.callbacks import Callback


class CallbackCollection(object):
    """Manage the collection of callbacks passed to EvoFlow.fit()
    """
    def __init__(self, model):
        self.callbacks = []
        self.model = model

    def add_callbacks(self, callbacks):

        if not isinstance(callbacks, list):
            callbacks = [callbacks]

        for cb in callbacks:
            if not isinstance(cb, Callback):
                raise TypeError('Callback must be subclass of Callback()')

        self.callbacks = callbacks

        # assign evoflow model to each callback
        for callback in callbacks:
            callback.set_model(self.model)

    def on_evolution_begin(self, populations):
        "Called at the start of the evolution"
        for callback in self.callbacks:
            callback.on_evolution_begin(populations)

    def on_evolution_end(self, populations):
        """Called at the end of the evolution"""

        for callback in self.callbacks:
            callback.on_evolution_end(populations)

    def on_generation_begin(self, generation):
        """Called at the start of a generation."""

        for callback in self.callbacks:
            callback.on_generation_begin(generation)

    def on_generation_end(self, generation, metrics, fitness_scores,
                          populations):
        """Called at the end of each generation.

        Args:
            generation (int): Generation index
            metrics ([type]): dictionnary containing user defined metrics
            fitness_scores (dict): dictionnary contraining the fitness scores
            populations (list): evolved populations
        """

        for callback in self.callbacks:
            callback.on_generation_end(generation, metrics, fitness_scores,
                                       populations)
