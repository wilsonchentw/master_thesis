#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import collections
import itertools
import multiprocessing
import os
import sys
import warnings

from svmutil import *
from liblinearutil import *
import numpy as np
import scipy
from sklearn.cluster import KMeans
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


def load_dataset(prefix):
    filename = prefix + ".npz"
    try:
        with np.load(filename) as fin:
            dataset = {name: fin[name] for name in fin}
    except IOError:
        dataset = preload_list(args.fin)
        label = dataset.pop('label', np.array([])).tolist()
        #np.savez_compressed(filename, **dataset)


def kmeans_bag_of_word(feature, dict_size):
    num_image = feature.shape[0]
    num_word = feature.shape[1]

    feature = feature.reshape(num_image * num_word, -1)
    codebook = KMeans(n_clusters=dict_size, copy_x=False, n_jobs=5)
    hard_code = codebook.fit_predict(feature).reshape(num_image, num_word)
    hard_code = hard_code.reshape(num_image, num_word)
    
    bow = np.zeros((num_image, dict_size))
    for idx, row in enumerate(hard_code):
        bow[idx] = np.bincount(row, minlength=dict_size)
        bow[idx] = bow[idx] / np.sum(bow[idx])
    return bow


if __name__ == "__main__":

    # Parse argument
    warnings.simplefilter(action="ignore", category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("fin", metavar="image_list", 
                        help="list with path followed by label")

    args = parser.parse_args()
    prefix = os.path.basename(args.fin).partition('.')[0]
    dataset = preload_list(args.fin)
    label = dataset.pop('label', np.array([])).tolist()

    hog = descriptor.extract_hog(dataset['path'])
    hog = np.sqrt(hog)
    train(label, hog.reshape(hog.shape[0], -1).tolist(), '-v 5 -q')

    #color = descriptor.extract_color(dataset['path'])
    #color = np.sqrt(color)
    #train(label, color.reshape(color.shape[0], -1).tolist(), '-v 5 -q')
