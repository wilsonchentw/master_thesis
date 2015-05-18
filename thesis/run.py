#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import warnings

import numpy as np

import descriptor
from util import *

def extract_from_scratch(filename, batchsize):
    dataset = {}
    with open(filename, 'r') as fin:
        for line_idx, line in enumerate(fin, 1):
            # Extract raw descriptor
            path, label = line.strip().split(' ')
            data = descriptor.extract_descriptor(label, path)

            # Update dataset
            for name, value in data.items():
                dataset.setdefault(name, []).append(value)

            # With each batch, print progress report
            if line_idx % batchsize == 0: 
                print "line {0} is done".format(line_idx)

    # Convert to numpy array for furthur usage
    for name in dataset:
        dataset[name] = np.array(dataset[name])

    return dataset


if __name__ == "__main__":

    # Parse argument
    warnings.simplefilter(action="ignore", category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("fin", metavar="image_list", 
                        help="list with path followed by label")

    args = parser.parse_args()
    title = args.fin.partition('.')[0]

    # Read dataset
    try:
        dataset = {}
        with np.load(title + ".npz", 'r') as fin:
            print "Load {0} from disk ... ".format(title + ".npz")
            for name in fin.files:
                dataset[name] = fin[name]
    except IOError:
        dataset = extract_from_scratch(args.fin, 100)
        np.savez_compressed(title, **dataset)
        for name in dataset:
            if name != 'label':
                filename = "{0}_{1}.dat".format(title, name)
                svm_write_problem(filename, dataset['label'], dataset[name])
