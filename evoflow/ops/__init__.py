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

# base
from ..engine.op import OP  # noqa: F401

# core
from .core import Dummy  # noqa: F401

# inputs
from ..engine.input import Input  # noqa: F401

# Shuffle
from .shuffle import Shuffle  # noqa: F401

# Reverse
from .reverse import Reverse1D  # noqa: F401
from .reverse import Reverse2D  # noqa: F401
from .reverse import Reverse3D  # noqa: F401

# mutation
from .random_mutation import RandomMutations1D  # noqa: F401
from .random_mutation import RandomMutations2D  # noqa: F401
from .random_mutation import RandomMutations3D  # noqa: F401

# crossover
from .uniform_crossover import UniformCrossover1D  # noqa: F401
from .uniform_crossover import UniformCrossover2D  # noqa: F401
from .uniform_crossover import UniformCrossover3D  # noqa: F401
from .single_crossover import SingleCrossover1D  # noqa: F401
from .single_crossover import SingleCrossover2D  # noqa: F401
from .single_crossover import SingleCrossover3D  # noqa: F401
from .dual_crossover import DualCrossover1D  # noqa: F401
from .dual_crossover import DualCrossover2D  # noqa:F401
from .dual_crossover import DualCrossover3D  # noqa: F401
