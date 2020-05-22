import json
from termcolor import cprint
import numpy as np
from tabulate import tabulate
from time import time
from pathlib import Path
from evoflow import __version__ as evoflow_version


class Logger():
    def __init__(self, system_info, backend):
        self.system_info = system_info
        self.ts = int(time())
        self.backend = backend
        if len(system_info['gpu']):
            gpu = system_info['gpu'][0]['name'].lower().replace(' ', '_')
        else:
            gpu = "CPU"

        result_dir = Path("results/%s" % (evoflow_version))
        if not result_dir.exists():
            result_dir.mkdir()

        fname = result_dir / ("%s_%s_%s.json" % (backend, gpu, self.ts))
        self.out = open(str(fname), 'w+')
        self.rows = []
        cprint('Bench results will be saved here:%s' % fname, 'green')

    def record_test(self, test_type, group, name, timings, num_runs,
                    num_generations, shape):
        shape = list(shape)
        record = {
            'ts': self.ts,
            'system': self.system_info,
            'backend': self.backend,
            'evoflow_version': evoflow_version,
            'test_type': test_type,
            'group': group,
            'name': name,
            'generations': num_generations,
            'num_runs': num_runs,
            'shape': shape,
        }
        record['input_size'] = int(np.prod(shape))
        record['timings'] = {
            "avg": float(np.average(timings)),
            "max": float(np.max(timings)),
            "min": float(np.min(timings)),
            "std": float(np.std(timings)),
            "raw": timings
        }
        self.out.write(json.dumps(record) + '\n')
        self.rows.append([
            group, name, shape,
            round(record['timings']['min'], 3),
            round(record['timings']['avg'], 3),
            round(record['timings']['max'], 3),
            round(record['timings']['std'], 3)
        ])

    def summary(self):
        print(
            tabulate(
                self.rows,
                headers=['group', 'name', 'shape', 'min', 'avg', 'max',
                         'std']))
