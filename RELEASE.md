# Changelog

## v0.4.1

* Benchmarking system implemented - see `/benchmark/benchmark_analysis.ipynb`

### Notable improvements and major changes

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
