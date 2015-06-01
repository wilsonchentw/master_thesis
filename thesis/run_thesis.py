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
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedKFold
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
    label = dataset.pop('label', np.array([]))
    folds = StratifiedKFold(label, n_folds=5, shuffle=True)

    from hog import raw_hog
    bin_grid = [32, 64, 128]
    block_grid = [16, 32, 64]
    for bins, block in itertools.product(bin_grid, block_grid):
        block = (block, block)
        get_hog = lambda image: raw_hog(image, bins, block, block)

        print "bins={0}, block={1}".format(bins, block)
        hog = descriptor.extract_descriptor(dataset['path'], get_hog)
        hog = hog.reshape(hog.shape[0], -1)

        acc = []
        for train_idx, test_idx in folds:
            train_hog, test_hog = hog[train_idx], hog[test_idx]
            train_label, test_label = label[train_idx], label[test_idx]

            gauss_nb = GaussianNB().fit(train_hog, train_label)
            acc.append(gauss_nb.score(test_hog, test_label))
        print "Cross Validation Accuracy = {0}%".format(np.mean(acc) * 100)
        train(label.tolist(), hog.tolist(), '-v 5 -q')
