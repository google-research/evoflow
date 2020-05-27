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
import altair as alt  # dymamic
from matplotlib.pylab import plt  # static graph
from tabulate import tabulate
import pandas as pd
import evoflow.backend as B
from evoflow.utils import unbox
from evoflow.io import print_debug


class Results(object):
    def __init__(self, debug=False):

        self.start_time = time()

        self._populations = None
        self._fitness_scores = None
        self._metrics_latest = defaultdict(dict)  # convinience holder
        self._metrics_history = defaultdict(lambda: defaultdict(list))
        self.debug = debug

    def get_populations(self):
        return unbox(self._populations)

    def set_population(self, populations):
        """Assign unboxed population

        Args:
        populations (list): list of population tensors

        Note: the whole class assumes populations as a list so don't set
        unboxed results or it will break everything.
        """

        if self.debug:
            print_debug(populations)

        self._populations = populations

    def display_populations(self,
                            top_k=10,
                            max_chromosome_len=1024,
                            rounding=None):

        for pop_idx, population in enumerate(self._populations):
            num_chromosomes = min(len(population), top_k)
            heatmap = []
            gene_max = 0
            for cidx, chromosome in enumerate(population):
                genes = []
                for gene in chromosome[:max_chromosome_len]:
                    if isinstance(rounding, type(None)):
                        genes.append(str(gene))
                    elif rounding > 0:
                        genes.append(round(float(gene), rounding))
                    else:
                        genes.append(int(gene))
                    gene_max = max(gene_max, gene)
                heatmap.append(genes)
                if cidx > num_chromosomes:
                    break

            fig, ax = plt.subplots()
            im = ax.imshow(heatmap, cmap="viridis", vmax=gene_max)

            # gradient bar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Gene value', rotation=-90, va="bottom")

            plt.title(f'Population - top {num_chromosomes}',
                      fontsize=16,
                      fontweight='bold')
            plt.xlabel("Genes")
            plt.ylabel("Chromosome")
            # fig.tight_layout()
            plt.show()

    def top_k(self, top_k=10, max_chromosome_len=20, precision=None):
        """Display the top k chromosome.

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

                if len(genes) != len(chromosome):
                    genes.append(' ...')

                genes = " ".join(genes)
                row = [self._fitness_scores[pop_idx][cidx], genes]
                if cidx == top_k:
                    break
                rows.append(row)
            print(
                tabulate(
                    rows,
                    headers=['fit score',
                             'genes [:%d]' % max_chromosome_len]))

    def plot_fitness(self, static=False):
        """Plots the various metrics"""
        if static:
            return self._build_plot_static('Fitness function')
        else:
            return self._build_plot('Fitness function')

    def plot(self, group_name, static=False):
        """Plots group metrics"""
        if static:
            return self._build_plot_static(group_name)
        else:
            return self._build_plot(group_name)

    def _build_plot_static(self, metric_group):
        metrics = self.get_metrics_history()
        if metric_group not in metrics:
            raise ValueError("can't find metric group")
        for name, values in metrics[metric_group].items():
            plt.plot(values, label=name)
        plt.legend(loc='best')
        plt.title(metric_group, fontsize=16, fontweight='bold')
        plt.xlabel("Generations")
        plt.show()

        return None

    def _build_plot(self, metric_group):
        "Build an altair plot for a given metric group"
        metrics = self.get_metrics_history()
        if metric_group not in metrics:
            raise ValueError("can't find metric group")

        data = metrics[metric_group]
        rows = []
        for name, values in data.items():
            for idx, val in enumerate(values):
                rows.append({'generation': idx, 'metric': name, 'value': val})
        metrics_pd = pd.DataFrame(rows)

        chart = alt.Chart(metrics_pd,
                          title='Fitness function').mark_line().encode(
                              x='generation', y='value',
                              color='metric').configure_title(fontSize=16,
                                                              offset=5,
                                                              orient='top',
                                                              anchor='middle')
        return chart

    def get_metrics_history(self):
        """Get the last evolution metrics values

        Returns:
            dict: name: value as list(float).
        """
        return self._metrics_history

    def get_latest_metrics(self, flatten=False):
        """Get the last evolution metrics values

        Args:
            flatten (bool, optional): Return metrics as a flat dictionary
            instead of a nested one

        Returns:
            dict: name:value as float.
        """

        if flatten:
            metrics = {}
            for group_name, group_data in self._metrics_latest.items():
                for metric, value in group_data.items():
                    k = "%s_%s" % (group_name, metric)
                    metrics[k] = value
            return metrics
        else:
            return self._metrics_latest

    def record_metrics(self, metrics):
        """Record metrics and track their history

        Args:
            metrics (dict(dict)): group of metrics to track. Of the form:
            [group][metric] = float(value)
        """
        for group, data in metrics.items():
            for metric, value in data.items():
                self._metrics_history[group][metric].append(value)
                self._metrics_latest[group][metric] = value

    def record_fitness(self, fitness_scores):
        """Compute and record fitness related metrics
        Args:
            fitness_scores (list(ndarray)): tensor holding fitness scores.
        """

        self._fitness_scores = fitness_scores

        # update history
        for pop_idx, fit_scores in enumerate(fitness_scores):
            METRICS = [['mean', B.mean], ['max', B.max], ['min', B.min]]

            for name, fn in METRICS:

                # only suffix is more than one population
                if len(fitness_scores) > 1:
                    name += "_%s" % pop_idx

                # compute result
                value = float(fn(fit_scores))

                self._metrics_history['Fitness function'][name].append(value)
                self._metrics_latest['Fitness function'][name] = value
