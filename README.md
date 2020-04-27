# GeneFlow - Evolutionary algorithms for humans

## Terminology

- **Gene**: atomic unit.
- **Chromosome**: ordered list of gene(s).
- **Genotype**: collection of chromosome(s).
- **Population of x**: collection of chromosomes or genotypes.
- **Generation**: One round of evolution. Equivalent to an epoch for DL.
- **Fitness function**: Function that evaluate how good is a given
chromosome. Equivalent to the loss function in deep learning except it doesn't
need to be differentiable.

### GeneFlow Terminolgy

- **evoluationary op**: Operation performed on a population of chromosome to
make them evolve. Common ops includes various type of Chromosomal crossovers
and Chromosomal mutations. Equivalent to deep-learning layers
(e.g a convolution layer).

- **evoluationary model**: Directed graph of evolutionary ops used to evolve
the population. Equivalent to a model architecture in deep-learning settings.

## Deep-learning versus Evoluationary algorithms.

Generally you want to use Deep-learning when the problem is continious/smooth
and evoluationary algorithms when the problem is discrete. For example voice
generation is smooth and solving (non-linear) equations is discrete.

## Disclaimer

This is not an official Google product.
