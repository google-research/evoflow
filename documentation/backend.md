# Backend information

## Adding an op

Adding a backend ops requires you to:

1. Implement the op for each backend using the same function prototype. It's
your responsibility to unify API accross the various underlying frameworks. As
a rule of thumb avoid exposing unecessary options or using positional argument
when not needed. This is done by adding code
to `evoflow/backend/[framework].py`

*Note*: Backend files are organized by ops category so make sure to put your op
in the correct place.

2. Add your function prototype to the backend loader so it is loaded by the
modules. This is done by adding it to `evoflow/backend/__init__.py`. Make sure
to add it at the same position than it is in the backend code.

3. Add at least one unit test in `tests/backend/test_[op catregory].py`. Make
sure to use the `backend` fixture to ensure all backend are properly tested.

4. Add your op to the benchmarking notebook `notebooks/benchmark.ipynb` so
   performance can be tracked


## Benchmarking
