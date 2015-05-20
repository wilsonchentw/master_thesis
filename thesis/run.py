#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import warnings

import numpy as np

import descriptor


def load_dataset(filename):
    try:
        with np.load(prefix + ".npz") as fin:
            dataset = {name: fin[name] for name in fin}
    except IOError:
        dataset = descriptor.extract_all(args.fin, batchsize=100)
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
    
