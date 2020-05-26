# EvoFlow performance

This page provide an overview of how we go about optimizing EvoFlow speed and the
current state of the various optimizations.

If you are interested in raw performance measurements or how the various
backend compares head to the [benchmark results analysis notebook](https://github.com/google-research/evoflow/blob/master/benchmark/benchmark_analysis.ipynb).

## Key type of optimizations

Here are the key optimization strategies that need to be pursued. Note that
some of those strategies (at least initially) build upon each other.

- **Per OP optimization**: Each OPs, Selection strategy and Fitness strategy must be optimizable
  via JIT and potentially XLA to ensure that EvoFlow is fast on single GPU/Machine

- **Batching / Multi GPU**: batching the populations will allows to do multi-gpu and have population
  larger than GPU ram. Requires each op to be 100% tf compatible.

- **Join JIT optimization**: Instead of compiling each part individually, we can compile the whole
  `evolve()` function as one. This requires all op to be JIT/XLA optimizable and a large refactoring.

- **TPU/Distributed training**: Allows to use multiples TPU / Machine using the tensorflow distributed
  strategies. Requires Batching / Multi-GPU to be implemented.

### Current status

| strategy                          |                               status                                |
| --------------------------------- | :-----------------------------------------------------------------: |
| Per OP optimization               |  [Partial](https://github.com/google-research/evoflow/projects/1)   |
| Batching                          | [Not started](https://github.com/google-research/evoflow/issues/42) |
| multi-GPU                         | [Not started](https://github.com/google-research/evoflow/issues/43) |
| Evolution model joint optimzation | [Not started](https://github.com/google-research/evoflow/issues/50) |
| TPU support                       | [Not started](https://github.com/google-research/evoflow/issues/44) |

The next sections describe for each optimization strategy what need to be done to
go faster and, what is blocking further improvements.

## Per function optimization

### How optimizations are implemented in the OPs

Enabling optimization, providing the OP code supports it is as simple as declaring
support as OP class attribute. Evoflow will take care of applying them at runtime.

For example [Shuffle()](https://github.com/google-research/evoflow/blob/master/evoflow/ops/shuffle.py)
declare it supports both *autograph* and *xla* by setting the variable `O_AUTOGRAPH`
and `O_XLA` to true:

```python
class Shuffle(OP):

    O_AUTOGRAPH = True
    O_XLA = True
```

### How to check optimization support

Checking what type of optimizations are supported for a given OP
is done using: `optimization_support`. For example:

```python
from evoflow.utils import optimization_support
from evoflow.ops import Shuffle
optimization_support(Shuffle)
```

will return:

```python
{'optimization_level': 2, 'optimizations': {'autograph': True, 'xla': True}}
```

Testing optimization support
`optimization_support`
`op_optimization_benchmark`

### How to benchmark optimizations

Testing the speedup provided by various optimization level is done using the
`op_optimization_benchmark()` function. For example to benchmark `Shuffle()`
you can use the following code:

```python
from evoflow.ops import Shuffle
from evoflow.utils import op_optimization_benchmark
NUM_RUNS = 10
pop_shape = (1000, 1000, 100)
population = B.randint(0, 256, pop_shape)
population_fraction = 0.5

OP = Shuffle(population_fraction)
counters = op_optimization_benchmark(population, OP, NUM_RUNS)
counters.report()
```

Counters are `PerfCounters()` ones so you can export them in other format.

### How to benchmark current ops

Speedup micro-benchmark for all the OPs are implemented directly in the op
files themselves so can be benchmark any op by calling the op file directly
from commandline - For example to benchmark `RandomMutation()` optimization you
will call `python.exe .\evoflow\ops\random_mutation.py`

## OPs optmizations current status and bottleneck

Here is a summary of EvoFlow support for autograph and XLA per OP, a rough
estimate of the speed-up provided and the tradeoff/ideas on how to push further.

| OP               |   Status   | autograph | +XLA  | Key bottleneck                         |
| ---------------- | :--------: | :-------: | :---: | -------------------------------------- |
| RandomMutation   | 🐢:disabled |   0.3x    |   ☠️   | `assign()` is slow and prevent XLA     |
| Shuffle          |   🐇:good   |    2x     |  16x  | can improve if tf.shuffle support axis |
| Reverse          | 🐢:disabled |   0.1x    |   ☠️   | `assign()` is slow and prevent XLA     |
| SingleCrossover  | 🐢:disabled |   0.2x    |   ☠️   | `assign()` is slow and prevent XLA     |
| DualCrossover    | 🐢:disabled |   0.1x    |   ☠️   | `assign()` is slow and prevent XLA     |
| UniformCrossover | 🐢:disabled |   0.2x    |   ☠️   | `assign()` is slow and prevent XLA     |

The reported measurements and bugs were made on EvoFlow runing on a RTX 2080ti,
TensorFlow 2.2.0, CUDA 10.1 on Windows 10.
