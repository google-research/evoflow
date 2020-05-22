from evoflow.ops import Shuffle


def bench_shuffle(population):
    return Shuffle(1)(population)
