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
from perfcounters import PerfCounters
from tqdm.auto import tqdm
import inspect


def slices2array(slices):
    "convert slices into an array. Use for assign"
    return [[s.start, s.stop] for s in slices]


def box(a):
    "box single tensor into a list"
    if isinstance(a, list):
        return a
    else:
        return [a]


def unbox(a):
    "Unbox list with a single element"
    if isinstance(a, list) and len(a) == 1:
        return a[0]
    else:
        return a


def op_optimization_benchmark(population, op, num_runs, num_warmup=1):
    """Run OP with the different level of optimization supported and
    report results as PerfCounters counterts.

    Args:
        population (tensor): population to run the bench on
        op (OP): OP to be benchmarked
        num_runs (int): number of execution to perform.

    Returns:
        PerfCounters: performance counters for various optimization levels.
    """
    cprint('[warn-up]', 'yellow')
    os = optimization_support(op)
    max_level = os['optimization_level']
    cprint('Max optimization level %s' % (max_level), 'magenta')
    cprint('Optimization status:%s' % os['optimizations'], 'blue')

    cprint('[%s micro benchmark]' % str(op.__class__.__name__), 'yellow')

    total_tests = (num_warmup + num_runs) * (max_level + 1)
    pb = tqdm(total=total_tests, desc='running tests', unit='ops')
    cnts = PerfCounters()

    for level in range(max_level + 1):
        cname = 'Optimization level: %d' % level
        op.set_optimization_level(level)

        # warmup
        for _ in range(num_warmup):
            op(population)
            pb.update(1)

        # real test
        cnts.start(cname)
        for _ in range(num_runs):
            op(population)
            pb.update(1)
        cnts.stop(cname)

    pb.close()
    return cnts


def optimization_support(op):
    """Return the level of optimization supported by a given op and
       which optimizations are supported

    Args:
        op (OP): the op to analyze

    Returns:
        dict: optimization info.
    """

    optims = {}
    for m in inspect.getmembers(op):
        if 'O_' in m[0]:
            optims[m[0]] = m[1]

    autograph = optims.get('O_AUTOGRAPH', False)
    xla = optims.get('O_XLA', False)

    if autograph and xla:
        max_level = 2
    elif autograph:
        max_level = 1
    else:
        max_level = 0

    return {
        'optimization_level': max_level,
        'optimizations': {
            'autograph': autograph,
            'xla': xla
        }
    }
