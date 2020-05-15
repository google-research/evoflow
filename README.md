# EvoFlow - Evolutionary algorithms for humans
![TensorFlow](https://github.com/google-research/evoflow/workflows/TensorFlow/badge.svg) ![Numpy](https://github.com/google-research/evoflow/workflows/Numpy/badge.svg)


## Install

`pip install evoflow`

## Deep-learning versus Evoluationary algorithms

Generally you want to use Deep-learning when the problem is continious/smooth
and evoluationary algorithms when the problem is discrete. For example voice
generation is smooth and solving (non-linear) equations is discrete.


## Terminology

- **Gene**: atomic unit. Equivalent to a neuron in deep-learning.
- **Chromosome**: ordered list of gene(s).
- **Genotype**: collection of chromosome(s). Used when the problem requires to
maximizes multiples fitness function at once.
- **Population of x**: collection of chromosomes or genotypes.
  That is what makes a Tensor.
- **Generation**: One round of evolution. Equivalent to an epoch in deep-learning.
- **Fitness function**: Function that evaluate how good/fit a given chromosome is.
  this is equivalent to the loss function in deep learning except it doesn't
need to be differentiable and aim to be maximized.

### EvoFlow Terminology

- **evoluationary op**: Operation performed on a population of chromosome to
make them evolve. Common ops includes various type of Chromosomal crossovers
and Chromosomal mutations. Equivalent to deep-learning layers
(e.g a convolution layer).

- **evoluationary model**: Directed graph of evolutionary ops used to evolve
the population. Equivalent to a model architecture in deep-learning settings.

## Disclaimer

This is not an official Google product.
