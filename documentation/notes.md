# Note from the architect

## Gotchas

- Tensorflow don't do shuffle in place accordingly to be cross-compatible
  the various backends ops including `B.shuffle()`, `B.full_shuffle()` are also
  not in place. That might caught you by surprise if you come from `numpy`.

- Tensorflow don't support fancy indexing so instead you need to use `B.assign`.

- use `floatx()`, `set_floatx()` and `intx()`, `set_intx()` to select
  numerical precision. By default we use `float32` and `int32` as EvoFlow
  focus on hardware acceleration. Coming from older frameworks you might
  expect `float64`.
