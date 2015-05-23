#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import collections
import multiprocessing
import os
import warnings

from svmutil import *
from liblinearutil import *
import numpy as np
import scipy
import sklearn

import descriptor


def preload_list(filename):
    with open(filename, 'r') as fin:
        dataset = collections.defaultdict(list)
        for line in fin:
            path, label = line.strip().split(" ")
            dataset['path'].append(path)
            dataset['label'].append(int(label))

        dataset['label'] = np.array(dataset['label'])
        return dataset


def grid_parameter(label, inst):
    s_grid = [0, 1, 3]
    c_grid = [100, 10, 1, 0.1, 0.01, 0.001, 1e-4, 1e-5]

    inst = inst.tolist()
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

    # Extract descriptor
    """
    filename = prefix + ".npz"
    try:
        with np.load(filename) as fin:
            dataset = {name: fin[name] for name in fin}
    except IOError:
        dataset = preload_list(args.fin)
        dataset['hog'] = descriptor.extract_hog(dataset['path'])
        #np.savez_compressed(filename, **dataset)

    label = dataset.pop('label', np.array([])).tolist()
    #print grid_parameter(label, dataset['hog'])

    train(label, dataset['hog'].tolist(), '-v 5 -q')
    """

    from hog import raw_hog
    import itertools

    dataset = preload_list(args.fin)
    label = dataset.pop('label', np.array([])).tolist()
 
    def special_hog(bins, block, step):
        get_hog = lambda x: raw_hog(x, bins, block, step).reshape(-1)
        hog = descriptor.extract_descriptor(dataset['path'], get_hog, None)
        acc = train(label, hog.tolist(), '-v 5 -q')
        print (acc, bins, block, step)

    bin_grid = [8, 16, 32, 64, 128]
    block_grid = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256)]
    step_grid = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
    pool = multiprocessing.Pool(10)
    for bins, block, step in itertools.product(bin_grid, block_grid, step_grid):
        if min(block) >= min(step) and min(step) * 4 >= max(block):
            pool.apply_async(special_hog, [bins, block, step])

    pool.close()
    pool.join()
