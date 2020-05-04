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

from termcolor import cprint


def arr2d():
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_shuffle_axis0(backends):
    for B in backends:
        t = B.tensor(arr2d())
        B.shuffle(t)
        for col in t:
            assert col[0] in [1, 4, 7]
            assert col[1] in [2, 5, 8]
            assert col[2] in [3, 6, 9]


def test_shuffle_axis1(backends):
    for B in backends:
        t = B.tensor(arr2d())
        cprint(t, 'blue')
        B.shuffle(t, axis=1)
        cprint(t, 'green')

        assert t[0][0] in [1, 2, 3]
        assert t[1][0] in [4, 5, 6]
        assert t[2][0] in [7, 8, 9]


def test_full_shuffle(backends):
    for B in backends:
        t = B.tensor(arr2d())
        cprint(t, 'blue')
        B.full_shuffle(t)
        cprint(t, 'green')

        ok = False

        # give it multiple try as identity is a valid shuffle
        for _ in range(100):
            if (t[0][0] != 1 or t[1][0] != 4) and (t[0][1] != 2
                                                   or t[1][1] != 5):  # noqa
                ok = True
                break
        assert ok


def test_randint_1D(backends):
    for B in backends:
        t = B.randint(0, 11, shape=10)
        assert t.shape == (10, )
        assert B.max(t) <= 10
        assert B.min(t) >= 0


def test_randint_2SD(backends):
    for B in backends:
        t = B.randint(0, 11, shape=(10, 20))
        assert t.shape == (10, 20)
        assert B.max(t) <= 10
        assert B.min(t) >= 0