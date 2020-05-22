import json
import tensorflow as tf
from collections import defaultdict
import numpy as np

import evoflow.backend as B
from evoflow.engine import EvoFlow
from evoflow.engine import FitnessFunction
from evoflow.selection import SelectFittest
from evoflow.population import uniform_population
from evoflow.ops import Input, Shuffle, Reverse1D


class TSPFitness(FitnessFunction):
    def __init__(self,
                 distances,
                 num_cities,
                 baseline_distance=0,
                 penality=100000,
                 **kwargs):
        """
        """
        self.num_cities = num_cities
        self.distances = B.flatten(distances)
        self.penality = penality
        self.baseline_distance = int(baseline_distance)
        super(TSPFitness, self).__init__(**kwargs)

    def call(self, population, normalize=True):
        """
            Parallel lookup and distance computation:
            - multiply the tensor by population shifed by 1 which gives
            the id to lookup in the flat
            distance array
            - reduce_sum for the total distance
            - 1/reduce_sum so fitness goes from 0 to 1
        """

        shifted_population = B.roll(population, 1, axis=1)
        idxs = (population * self.num_cities) + shifted_population
        distances = B.take(self.distances, idxs)

        # total distance
        total_distance = B.sum(distances, axis=1)
        return total_distance


def tsp_setup(num_cities):

    # get files
    zip_fname = "tsp_%s.zip" % num_cities
    origin = "https://storage.googleapis.com/evoflow/datasets/tsp/cities_%s.zip" % num_cities  # noqa
    download_path = tf.keras.utils.get_file(zip_fname, origin, extract=True)

    # process city info
    json_fname = "%s/cities_%s.json" % (download_path.replace(zip_fname,
                                                              ''), num_cities)
    cities = json.loads(open(json_fname).read())

    idx2city = {}
    for city in cities:
        idx2city[city['idx']] = city

    chart_data = defaultdict(list)
    for city in cities:
        chart_data['lat'].append(city['lat'])
        chart_data['lon'].append(city['lon'])
        chart_data['name'].append(city['name'])
        chart_data['population'].append(city['population'])

    distance_fname = "%sdistances_%s.npz" % (download_path.replace(
        zip_fname, ''), num_cities)
    distances = np.load(distance_fname)['distances']
    distances = distances.astype(B.intx())
    return cities, chart_data, distances, idx2city


def test_solve_tsp():
    population_shape = (100, 10)
    generations = 30

    cities, chart_data, distances, idx2city = tsp_setup(10)
    BASELINE = 6000
    NUM_CITIES = len(cities)
    NUM_REVERSE_OPERATIONS = 4
    MAX_REVERSE_PROBABILITY = 0.3
    REVERSE_POPULATION_FRACTION = 0.3
    MIN_REVERSE_PROBABILITY = 0.1

    SHUFFLE_POPULATION_FRACTION = 0.2

    population = uniform_population(population_shape)
    rpi = MAX_REVERSE_PROBABILITY / NUM_REVERSE_OPERATIONS
    reverse_probabilty = 1 - rpi

    # Evolution model

    inputs = Input(shape=population.shape)
    x = inputs
    for idx in range(NUM_REVERSE_OPERATIONS):
        x = Reverse1D(population_fraction=REVERSE_POPULATION_FRACTION,
                      max_reverse_probability=reverse_probabilty)(x)
        reverse_probabilty = max(reverse_probabilty - rpi,
                                 MIN_REVERSE_PROBABILITY)

    x = Shuffle(population_fraction=SHUFFLE_POPULATION_FRACTION)(x)
    outputs = x
    ef = EvoFlow(inputs, outputs, debug=False)

    evolution_strategy = SelectFittest(mode='min')
    fitness_fn = TSPFitness(distances, NUM_CITIES)
    ef.compile(evolution_strategy, fitness_fn)
    results = ef.evolve(population, generations=generations, verbose=0)
    metrics = results.get_latest_metrics()
    assert metrics['Fitness function']['min'] < BASELINE
