# Changelog

## v0.3.1

### New features

* Added roll() operator
* Added Shufle() OP
* Added Reverse() OP

### Key improvements and changes

* SelectFitest now allows to select individuals with the lowest fitness value.
* RandomMutation now clip to max_value - 1 to be consistent with RandInt()


## v0.3

### New features

* Tensorflow backend is working and the defaults.
* Custom metrics can now be recorded in fitness function.

### Key improvements and changes

* Many Backend ops signature changed to work with Tensorflow
* Plotly integration complete

## v0.2

* Ops refactored to support 1D, 2D and 3D tensors seamlessly.

## v0.1

* Initial function release.
