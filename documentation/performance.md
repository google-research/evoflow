# EvoFlow performance

This page provide an overview of how we go about optimizing EvoFlow speed and the
current state of the various optimizations. If you are interested in raw
performance measurements or how the various backend compares head to the
benchmark analysis notebook.

## how to make EvoFlow fast?

Here are the few key optimization strategies that need to be pursued. Note that
some of those strategies (at least initially) build upon each other.

- **JIT optimization**: Each OPs, Selection strategy and Fitness strategy must be optimizable
  via JIT and potentially XLA. This ensure EvoFlow is fast on single GPU/Machine

- **Batching / Multi GPU**: batching the populations will allows to do multi-gpu and have population
  larger than GPU ram. Requires each op to be 100% tf compatible.

- **Join JIT optimization**: Instead of compiling each part individually, we can compile the whole
  `evolve()` function as one. This requires all op to be JIT/XLA optimizable and a large refactoring.

- **TPU/Distributed training**: Allows to use multiples TPU / Machine using the tensorflow distributed
  strategies. Requires Batching / Multi-GPU to be implemented.

### Current status

| strategy               |                               status                                |
| ---------------------- | :-----------------------------------------------------------------: |
| JIT                    |  [Partial](https://github.com/google-research/evoflow/projects/1)   |
| Batching               | [Not started](https://github.com/google-research/evoflow/issues/42) |
| multi-GPU              | [Not started](https://github.com/google-research/evoflow/issues/43) |
| Joint JIT optimziation | [Not started](https://github.com/google-research/evoflow/issues/50) |
| TPU support            | [Not started](https://github.com/google-research/evoflow/issues/44) |

The next sections describe for each optimization strategy what need to be done to
go faster and, what is blocking further improvements.

## OPs optmizations status and current speedup

Here is a summary of EvoFlow support for autograph and XLA per OP, a rough
estimate of the speed-up provided and the tradeoff/ideas on how to push further.

| OP              | autograph | +XLA speedup | notes                                  |
| --------------- | :-------: | :----------: | -------------------------------------- |
| RandomMutation  |    1x     |     800x     | Assign can be improved                 |
| Shuffle         |    2x     |     20x      | can improve if tf.shuffle support axis |
| Reverse         |    5x     |    4000x     | Assign can be improved                 |
| SingleCrossover |    70x    |     80x      | Assign and Full shuffle                |
| DualCrossover   |    70x    |     110x     | Assign                                 |
| DualCrossover   |    70x    |     110x     | can't random size of the crossover     |
The reported speedup measurements were made on a RTX 2080ti with
TensorFlow 2.2.0, CUDA 10.1 on Windows 10.

### Reproducing speedup numbers

Speedup micro-benchmark are implemented in the op file itself and can be run by
calling it diretly with python. For example to test RandomMutation speedup you
can call: `python.exe .\evoflow\ops\random_mutation.py`s