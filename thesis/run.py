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
        for line_idx, line in enumerate(fin):
            path, label = line.strip().split(' ')
            data = Image(path, label)

            # Extract feature
            data = feature.extract_feature(data)
            dataset.append(data)

            # Progress report
            batch_size = 1
            if (line_idx + 1) % batch_size == 0: 
                print "line {0} is done".format(line_idx)
                break

    # Save for libsvm format
    title = args.fin.partition('.')[0]
    descriptor = {
        'hog': (data.hog for data in dataset), 
        'phog': (data.phog for data in dataset), 
        'color': (data.color for data in dataset), 
        'gabor': (data.gabor for data in dataset), 
    }

    #svm_write_problem("filename", label, color)
