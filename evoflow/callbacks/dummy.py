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
from .callback import Callback


class DummyCallback(Callback):
    "Dummy callback used for testing"

    def on_evolution_begin(self, populations):
        assert isinstance(populations, list)
        for pop in populations:
            assert B.is_tensor(pop)

    def on_evolution_end(self, populations):
        assert isinstance(populations, list)
        for pop in populations:
            assert B.is_tensor(pop)

    def on_generation_begin(self, generation):
        assert isinstance(generation, int)

    def on_generation_end(self, generation, metrics, fitness_scores,
                          populations):
        assert isinstance(generation, int)
        assert isinstance(metrics, dict)
        assert isinstance(fitness_scores, dict)
        assert isinstance(populations, list)
        for pop in populations:
            assert B.is_tensor(pop)
