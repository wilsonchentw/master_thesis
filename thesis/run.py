#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
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
        dataset = descriptor.extract_all(args.fin, batchsize=5)
        np.savez_compressed(prefix, **dataset)

    return dataset

if __name__ == "__main__":

    # Parse argument
    warnings.simplefilter(action="ignore", category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("fin", metavar="image_list", 
                        help="list with path followed by label")

    args = parser.parse_args()
    prefix = os.path.basename(args.fin).partition('.')[0]
    dataset = load_dataset(args.fin)

    label = dataset.pop('label', None).tolist()
    for name in dataset:
        print "{0}: ".format(name)
        inst = dataset[name].tolist()

        c = [10, 1, 0.1, 0.01, 0.001]
        s = [1, 3, 4, 5, 6, 7]
        for s, c in itertools.product(s, c):
            option = "-s {0} -c {1} -v 5 -q".format(s, c)
            print option
            model = train(label, inst, option)
