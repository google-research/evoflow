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

from geneflow.fitness import CosineSimilarity
import geneflow.backend as B


def test_python_input(backends):
    reference = [2, 0, 1, 1, 0, 2, 1, 1]

    r = B.tensor(reference)
    cs = CosineSimilarity(reference)  # python input
    assert int(cs(r)) == 1


def test_cosine_similarity_single(backends):
    reference = [2, 0, 1, 1, 0, 2, 1, 1]
    different = [2, 1, 1, 0, 1, 1, 1, 1]
    expected_distance_diff = round(0.8215838362577491, 3)

    r = B.tensor(reference)
    d = B.tensor(different)
    print('r', type(r))
    print('d', type(d))
    cs = CosineSimilarity(r)

    # similar vector have a distance of 1
    distance = cs(r)
    print('cs', type(cs))
    print('distance', type(distance))

    assert int(distance) == 1

    # different vectors
    distance = cs(d)
    assert round(float(distance), 3) == expected_distance_diff


def test_cosine_similarity_population(backends):
    "test the function works on a population and broadcast correctly"

    reference = [2, 0, 1, 1, 0, 2, 1, 1]
    different = [2, 1, 1, 0, 1, 1, 1, 1]
    expected_distance_diff = round(0.8215838362577491, 3)
    r = B.tensor(reference)
    d = B.tensor(different)
    population = B.tensor([r, d, r])
    cs = CosineSimilarity(r)

    # similar vector have a distance of 1
    distances = cs(population)

    assert int(distances[0]) == 1
    assert round(float(distances[1]), 3) == expected_distance_diff
    assert int(distances[2]) == 1
