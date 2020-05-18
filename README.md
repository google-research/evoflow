# EvoFlow - Evolutionary algorithms for humans
![TensorFlow](https://github.com/google-research/evoflow/workflows/TensorFlow/badge.svg) ![Numpy](https://github.com/google-research/evoflow/workflows/Numpy/badge.svg)

## You have just found EvoFlow

EvoFlow is a modern hardware accelerated genetic algorithm framework that recast
genetic algorithm programing as a dataflow computation on tensors.
Conceptually is very similar to how Tensorflow & Keras are approaching
deep-learning so if you have experience with any of those you will feel right
at home.

Under the hood, EvoFlow leverage Tensorflow or Cupy to provide hardware
accelerated genetic operations. If you don't have a GPU, you can run EvoFlow on
Google Colab or it will just work fine on your CPU.

## Getting started in 30 seconds

1. Install EvoFlow: `pip install evoflow`
2. Head to our [hello world notebook](https://github.com/google-research/evoflow/blob/master/notebooks/maxone.ipynb) that will shows you how to use EvoFlow to solve the classic MaxOne problem.

## Tutorials

The following tutorials are availables

| Problem                 | Description                                                 | Key concepts showcased                                                                                                                                             |
| ----------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| MaxOne                  | Maximize the number of ones in a chromosome                 | <ul><li>`EvoFlow` core API</li><li>`RandomMutation` OP</li><li> `UniformCrossOver` Op</li><li>Evolution model construction</li><li>`Results` basic usage</li></ul> |
| Travel Salesman problem | Visit each city once while minimizing the distance traveled | <ul><li>Custom `Fitness function`</li><li>Genes permuting Ops: `Shuffle` and `Reverse`</li><li>Evolution model programatic construction</li></ul>                  |

Genetic Algorithm are used to solve a [wide variety of problems](https://en.wikipedia.org/wiki/List_of_genetic_algorithm_applications)

## Deep-learning versus Evoluationary algorithms

Generally you want to use Deep-learning when the problem is continious/smooth
and evoluationary algorithms when the problem is discrete. For example voice
generation is smooth and solving (non-linear) equations is discrete.

Concretely this means that the fitness functions you use to express what constraint
to solve are very similar to the loss functions in deep-learning except they
don't need to be differentiable and therefore can perform arbitrary computation.

However the cost of fitness function increased expressiveness and flexibility
compared to neural network loss is that we don't have the gradients to help
guide the model convergence and therefore coverging is more computationaly
expensive which is why having a hardware accelerated framework is essential.

## Genetic Algorithm terminology

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

- **Evoluation op**: Operation performed on a population of chromosome to
make them evolve. Common ops includes various type of Chromosomal crossovers
and Chromosomal mutations. Equivalent to deep-learning layers
(e.g a convolution layer).

- **Evolution model**: Directed graph of evolutionary ops that is used
  to evolve the population. This is equivalent to a model architecture
  in deep-learning settings.

## Disclaimer

This is not an official Google product.
