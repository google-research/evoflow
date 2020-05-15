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

from evoflow import io


def test_print_debug_multi_msg():
    io.print_debug('component', 'msg1', 'msg2')


def test_casting_msg():
    io.print_debug('component', [1, 2], 3)


def test_print_debug_single_msg():
    io.print_debug('component', "simple")
