#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import warnings

import numpy as np

import feature
from util import *


class Image(object):
    """ Store path, label, and extracted descriptors """
    def __init__(self, path, label):
        self.path = path
        self.label = label


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("fin", metavar="image_list", 
                        help="list with path followed by label")

    # Parsing image list
    args = parser.parse_args()
    dataset = []
    with open(args.fin, 'r') as fin:
        for line_idx, line in enumerate(fin, 1):
            path, label = line.strip().split(' ')

            # Extract features
            data = Image(path, label)
            feature.extract_all(data)
            dataset.append(data)

            if (line_idx % 1) == 0:
                print "line {0} done".format(line_idx)
                break

    for 
    
