# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import time
from collections import defaultdict
import matplotlib.pyplot as plt


from tabulate import tabulate

import geneflow.backend as B
from geneflow.utils import unbox


class Results(object):

    def __init__(self):
        self._metrics_latest = {}  # convinience holder
        self._metrics_history = defaultdict(list)

        self.start_time = time()

        self._populations = None
        self._fitness_scores = None

    def get_populations(self):
        return unbox(self._populations)

    def set_population(self, populations):
        """Assign unboxed population

        Args:
        populations (list): list of population tensors

        Note: the whole class assumes populations as a list so don't set
        unboxed results or it will break everything.
        """
        self._populations = populations

    def display_populations(self, top_k=10, max_chromosome_len=20,
                            precision=None):
        """Display the population

        Args:
            top_k (int, optional): Number of chromosomes to display.
            Defaults to 10.

            expected_max_value (int, optional): how many gene to display per
            chromosomes.

            precision (int, optional): how many digits per chromosome to
            display. Default to None. If None full value.

        FIXME: use HTML while in notebook
        """

        for pop_idx, population in enumerate(self._populations):
            rows = []
            for cidx, chromosome in enumerate(population):
                genes = []
                for gene in chromosome[:max_chromosome_len]:
                    if isinstance(precision, type(None)):
                        genes.append(str(gene))
                    elif precision > 0:
                        genes.append(str(round(float(gene), precision)))
                    else:
                        genes.append(str(int(gene)))

                genes = " ".join(genes)
                row = [self._fitness_scores[pop_idx][cidx], genes]
                if cidx == top_k:
                    break
                rows.append(row)
            print(tabulate(rows, headers=['fit score', 'genes']))

    def plot_metrics(self):
        """Plots the various metrics"""
        metrics = self.get_metrics_history()
        for row_id, name in enumerate(sorted(metrics.keys())):
                plt.figure(row_id + 1)
                title = name.replace('_', ' ').capitalize()
                plt.title(title)
                plt.plot(metrics[name])

    def get_metrics_history(self):
        """Get the last evolution metrics values

        Returns:
            dict: name: value as list(float).
        """
        return self._metrics_history

    def get_latest_metrics(self):
        """Get the last evolution metrics values

        Returns:
            dict: name:value as float.
        """
        return self._metrics_latest

    def record_fitness(self, fitness_scores):
        """Compute and record fitness related metrics
        Args:
            fitness_scores (list(ndarray)): tensor holding fitness scores.
        """

        self._fitness_scores = fitness_scores

        # update history
        for pop_idx, fit_scores in enumerate(fitness_scores):
            METRICS = [
                ['mean', B.mean],
                ['max', B.max]
            ]

            for stem, fn in METRICS:
                name = "fitness_%s" % stem

                # only suffix is more than one population
                if len(fitness_scores) > 1:
                    name += "_%s" % pop_idx

                # compute result
                value = float(fn(fit_scores))

                self._metrics_history[name].append(value)
                self._metrics_latest[name] = value
