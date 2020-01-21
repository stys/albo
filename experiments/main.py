#!/usr/bin/env python3

import numpy as np

import sys
import re
import imp
import logging

from os import makedirs
from os.path import join as join_path

from argparse import ArgumentParser
from pyhocon import ConfigFactory, ConfigTree


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', level=logging.DEBUG,
                        datefmt='%Y-%m-%dT%H:%M:%S')

    pattern = re.compile('-D(.*)=(.*)')
    conf_override = dict()
    argv_filtered = []
    for a in sys.argv:
        m = pattern.match(a)
        if m is not None:
            conf_override[m.group(1)] = m.group(2)
        else:
            argv_filtered.append(a)

    parser = ArgumentParser()
    parser.add_argument('--conf', required=True, help='Path to experiment config file')
    parser.add_argument('--task', required=True, help='Path to task file')
    parser.add_argument('--dir', required=True, help='Path to output dir')
    parser.add_argument('--force-makedirs', action='store_true')
    parser.add_argument('--nruns', type=int, default=1, help='Repeat task many times')
    args, other = parser.parse_known_args(argv_filtered)

    conf = ConfigFactory.parse_file(args.conf)
    conf_override = ConfigFactory.from_dict(conf_override)
    conf_merged = ConfigTree.merge_configs(conf, conf_override)

    task_module = imp.load_source('task', args.task)
    task_generator = task_module.TaskGenerator(conf=conf_merged)

    makedirs(args.dir, exist_ok=args.force_makedirs)
    for task in task_generator.generate():
        task_dir = join_path(args.dir, task.key)
        makedirs(task_dir, exist_ok=args.force_makedirs)

        for i in range(args.nruns):
            logging.info('Running task %s run %d', task.key, i)
            work_dir = join_path(task_dir, 'run_%03d' % i)
            makedirs(work_dir, exist_ok=args.force_makedirs)

            x, y, trace = task.run(print_file=None, verbose=True)

            trace_file_name = join_path(work_dir, 'trace')
            np.save(trace_file_name, trace)
