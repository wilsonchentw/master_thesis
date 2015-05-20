#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import warnings

from svmutil import *
from liblinearutil import *
import numpy as np
import sklearn

def load_dataset(filename):
    try:
        with np.load(prefix + ".npz") as fin:
            dataset = {name: fin[name] for name in fin}
    except IOError:
        #dataset = descriptor.extract_all(args.fin, 100)
        #np.savez_compressed(prefix, **dataset)

    return dataset

if __name__ == "__main__":

    # Parse argument
    warnings.simplefilter(action="ignore", category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("fin", metavar="image_list", 
                        help="list with path followed by label")

    args = parser.parse_args()
    prefix = os.path.basename(args.fin).partition('.')[0]
    dataset = load_dataset(filename)

    #svm_write_problem(data_name, label, data[name])
 
