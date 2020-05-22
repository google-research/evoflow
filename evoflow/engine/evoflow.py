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

import networkx as nx
from termcolor import cprint
from tabulate import tabulate
from collections import defaultdict
from tqdm.auto import tqdm

from evoflow.utils import box
from evoflow.io import print_debug
from .results import Results

import tensorflow as tf


class EvoFlow(object):
    def __init__(self, inputs, outputs, debug=False):

        if not inputs:
            raise ValueError("Inputs can't be empty")
        if not outputs:
            raise ValueError("Ouputs can't be empty")

        # FIXME: check we have the same number of inputs and output
        # and they have the same shape because we are using the ouputs as
        # the next inputs

        # set debug
        self.debug = debug

        # graph underlying structure
        self.graph = nx.DiGraph()

        # tracking structures
        self.idx2op = {}  # op object
        self.idx2results = {}  # op computation result
        self.idx2input_idx = defaultdict(set)  # ops used as inputs
        self.idx2ouput_ops = defaultdict(set)  # ops used as outputs

        self.inputs_idx = []  # track which op idx are inputs
        self.outputs_idx = []  # track what op idx are outputs

        self.fitness = None
        self.compiled = False
        self._results = None

        # storing inputs tensors
        self.inputs = box(inputs)
        for ipt in self.inputs:
            self.inputs_idx.append(ipt.idx)

        # output
        self.outputs = box(outputs)
        for output in self.outputs:
            self.outputs_idx.append(output.idx)

        # build forward graph
        for output in self.outputs:
            self._add_op_to_graph(output, None, self.debug)

        # FIXME: check that the graph is fully connected from input to output

        # coerce exec_path as a list to allow reuse accros batches.
        self.execution_path = list(nx.topological_sort(self.graph))

    def compile(self, selection_strategy, fitness_functions):
        """Configure evoluationary model for training

        """
        # FIXME: check args validity
        self.selection_strategy = selection_strategy
        self.fitness_functions = box(fitness_functions)
        self.compiled = True

        self._results = Results(debug=self.debug)

    def history(self):
        return self._history

    def evolve(self, populations, generations=1, callbacks=None, verbose=1):

        if not self.compiled:
            raise ValueError("compile() must be run before using the graph")
            return

        populations = box(populations)
        self.print_debug("Initial Populations", populations)

        if not len(populations) == len(self.inputs):
            raise ValueError('The numbers of population must be equal\
                 to number of inputs')

        # assign initial value to inputs
        current_populations = []
        for pop_idx, ipt in enumerate(self.inputs):
            self.inputs[pop_idx].assign(populations[pop_idx])
            pop = self.inputs[pop_idx].get()
            current_populations.append(pop)
        num_populations = len(current_populations)
        self.print_debug('Initial current_populations', current_populations)

        # progress bar
        if verbose:
            pb = tqdm(total=generations, unit='generation')

        # evolve loop
        for generation_idx in range(generations):

            # callbacks
            if callbacks:
                for callback in callbacks:
                    callback.on_generation_begin(generation_idx)

            # perform evolution
            evolved_populations = self.perform_evolution()

            # assign evolved populations
            self.print_debug(generation_idx, 'evolved pop',
                             evolved_populations)

            fitness_scores_list = []  # keep track of fitness score accros pops
            for pop_idx in range(num_populations):
                # find current informaiton
                current_pop = current_populations[pop_idx]
                evolved_pop = evolved_populations[pop_idx]
                fitness_function = self.fitness_functions[pop_idx]

                self.print_debug('current_population', pop_idx, current_pop)
                self.print_debug('evolved_population', pop_idx, evolved_pop)
                # select population
                new_population, fitness_scores, metrics = self.selection_strategy(  # noqa
                    fitness_function, current_pop, evolved_pop)

                # update metrics
                self._results.record_metrics(metrics)

                # update population tensor
                self.inputs[pop_idx].assign(new_population)

                # track current population
                current_populations[pop_idx] = new_population
                # record fitness scores
                fitness_scores_list.append(fitness_scores)

            self._results.record_fitness(fitness_scores_list)

            latest_metrics = self._results.get_latest_metrics(flatten=True)

            # callbacks
            if callbacks:
                for callback in callbacks:
                    callback.on_generation_end(generation_idx, latest_metrics,
                                               fitness_scores_list,
                                               populations)

            # progress bar
            if verbose:
                formatted_metrics = {}
                for name, value in latest_metrics.items():
                    name = name.lower().replace(' ', '_')
                    formatted_metrics[name] = value
                pb.set_postfix(formatted_metrics)
                pb.update()

        if verbose:
            pb.close()

        # record last population
        self._results.set_population(current_populations)

        # return last evolution
        return self._results

    def perform_evolution(self):
        """
        Evolve population

        Args:
            populations (list): populations to evolve.
        """

        # single batch # FIXME: move to a batch function as we need for
        # evaluate
        self.print_debug('execution path:', self.execution_path)

        for op_idx in self.execution_path:
            op = self.idx2op[op_idx]

            self.print_debug('|- %s(%s)' %
                             (op.idx, self.idx2input_idx[op.idx]))

            # fetching inputs values
            inputs = []
            for input_idx in self.idx2input_idx[op_idx]:
                inputs.append(self.idx2results[input_idx])
            self.idx2results[op_idx] = op._call_from_graph(inputs)

        self.print_debug('idx2results:', self.idx2results.keys())

        # collect results
        results = []

        for op_idx in self.outputs_idx:
            results.append(self.idx2results[op_idx])

        return results

    def summary(self):
        "print a summary of the data flow"
        rows = []

        for op_idx in self.execution_path:
            op = self.idx2op[op_idx]

            # inputs
            if len(self.idx2input_idx[op_idx]):
                inputs = ",".join([o for o in self.idx2input_idx[op_idx]])
            else:
                inputs = ''

            # the op id and its shape
            op_info = "%s (%s)" % (op.idx, op.op_type)

            # shape
            op_shape = "%s" % str(op.get_output_shapes())

            # output
            # if len(self.idx2ouput_ops[op_idx]):
            #     outputs = ",".join([o for o in self.idx2ouput_ops[op_idx]])  # noqa E501
            # else:
            #     outputs = '>>'

            rows.append([op_info, op_shape, inputs])
        print(tabulate(rows, headers=['OP (type)', 'Output Shape', 'Inputs']))

    def print_debug(self, *msg):
        "output debug message"
        if self.debug:
            print_debug('GeneFlow', *msg)

    def _add_op_to_graph(self, op, output_op=None, debug=0):
        """
        Recursively insert nodes in the DAG

        Take the opportunity to compute outbound nodes and various variables

        Args:
            op (Op): the operation to insert
            output_op (Op): the output_op to add.
            debug (int): print debug
        """

        self.idx2op[op.idx] = op
        self.graph.add_node(op.idx)

        # recording inbound op if any
        if output_op:
            self.idx2ouput_ops[op.idx].add(output_op.idx)

        # Reversing the graph and storing it - notice:in_op become op
        for in_op in op.input_ops:

            self.idx2input_idx[op.idx].add(in_op.idx)

            if debug:
                cprint("[edge] %s->%s" % (in_op.idx, op.idx), 'yellow')

            self.graph.add_edge(in_op.idx, op.idx)
            self._add_op_to_graph(in_op, op, self.debug)  # up the chain
