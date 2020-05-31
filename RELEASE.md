# Changelog

## v0.5.2

### Notable improvements and major changes

* Added Callback API

## v0.5.1

### Notable improvements and major changes

* Added static output for visualization so they show up in notebook on github
* Added partial population heatmap visualization (1D / static only)
* Move to numpy as default backend until tensorflow performance is resolved

## v0.5.0

This release focuses on optimizing single GPU/CPU performances to ensure
that using tensorflow provide compeling benefits compared to numpy.

### Notable improvements and major changes

* Added an Optimization dispatcher that allows ops that support it to leverage
  [TensorFlow optimization](https://www.tensorflow.org/api_docs/python/tf/function)
  and [XLA compilation](https://www.tensorflow.org/xla) if they support / benefit it.

* `RandomMutation()` now support TF and XLA optimization.

## v0.4.1

* Benchmarking system implemented - see `/benchmark/benchmark_analysis.ipynb`

### Notable improvements and major changes

* Moved from ploty to altair for results plotting for privacy reasons. Plotly
  graph being rendered on plotly servers. Also makes notebook smaller
  and rending faster.
* TSP notebook simplified and improved
* Many cupy and numpy backend bugs fixed

## v0.4

### New features

* Callbacks can now be added to the evolve loop

### New genetic operations

The new added ops are meant to allow to solve problem that requires to maintain
a list of unique values such as the travel saleman problem.

* `Shuffle()`: random permutations of the gene within each chromosome.
* `Reverse()`: reverse part of the genes inside the chromosme.

### New backend functions

The backend now offers the following functions:

* `roll()`
* `range()`
* `unique_with_count()`

### Notable improvements and major changes

* `SelectFitest()` now allows to select individuals with the lowest fitness value.
* `RandomMutation()` now clip to max_value - 1 to be consistent with `randInt()`
* `evolve` now support `verbose=0` to suppress progress bar. Useful to override
  training UI with callback.

As usual also a lot of bug fixes.

## v0.3

### New features

* Tensorflow backend is working and the defaults.
* Custom metrics can now be recorded in fitness function.

### Notable improvements and major changes

* Many Backend ops signature changed to work with Tensorflow
* Plotly integration complete

## v0.2

* Ops refactored to support 1D, 2D and 3D tensors seamlessly.

## v0.1

* Initial function release.
