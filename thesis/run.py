#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import collections
import itertools
import os
import warnings

from svmutil import *
from liblinearutil import *
import numpy as np
import scipy
import sklearn

import descriptor

def load_dataset(filename):
    try:
        with np.load(prefix + ".npz") as fin:
            dataset = {name: fin[name] for name in fin}
    except IOError:
        dataset = descriptor.extract_all(filename, 100, True)
        np.savez_compressed(prefix, **dataset)

    return dataset


def grid_parameter(label, inst):
    inst = inst.tolist()
    s_grid = [0, 1, 3]
    c_grid = [100, 10, 1, 0.1, 0.01, 0.001, 1e-4]

    acc = np.zeros((len(s_grid), len(c_grid)))
    for s_idx, s in enumerate(s_grid):
        for c_idx, c in enumerate(c_grid):
            option = "-s {0} -c {1} -v 5 -q".format(s, c)
            acc[s_idx, c_idx] = train(label, inst, option)
    return acc


if __name__ == "__main__":

    # Parse argument
    warnings.simplefilter(action="ignore", category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("fin", metavar="image_list", 
                        help="list with path followed by label")

    args = parser.parse_args()
    prefix = os.path.basename(args.fin).partition('.')[0]
    dataset = load_dataset(args.fin)
    label = dataset.pop('label', np.array([])).tolist()

    for name in dataset:
        acc = grid_parameter(label, dataset[name])
        print "{0}: ".format(name)
        print acc
