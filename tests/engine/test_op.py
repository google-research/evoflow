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
from evoflow.ops import Dummy, Input


def test_eager():
    "when passing concrete value GF is suppposed to return a concrete value"
    val = B.tensor(42)
    assert Dummy(debug=True)(val) == val


def test_graph():
    "When passing OPs GF is supposed to return a graph"
    i = Input((42, 1))
    d1 = Dummy()(i)
    d2 = Dummy()(i)
    r = Dummy()([d1, d2])
    assert issubclass(type(r), Dummy)
    assert r.call(42) == 42  # explicitly execute the graph op
