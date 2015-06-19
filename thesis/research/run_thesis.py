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
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
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


def bag_of_word(feature, dict_size):
    num_image = feature.shape[0]

    codebook = KMeans(n_clusters=dict_size, copy_x=False, n_jobs=5)
    codebook.fit(np.concatenate(feature))

    bow = np.zeros((num_image, dict_size))
    for idx, f in enumerate(feature):
        encode = codebook.predict(feature[idx])
        bow[idx] = np.bincount(encode, minlength=dict_size)
        bow[idx] = bow[idx].astype(float) / np.sum(bow[idx])

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

    hog = descriptor.extract_hog(dataset['path'])
    hog = hog.reshape(hog.shape[0], -1)
    for train_idx, test_idx in folds:
        train_hog, test_hog = hog[train_idx], hog[test_idx]
        train_label, test_label = label[train_idx], label[test_idx]
        model = train(train_label.tolist(), train_hog.tolist(), '-q')
        pred, acc, v = predict(test_label.tolist(), test_hog.tolist(), model)

        names = [
            'Croissants', 'Chasiu', 'Stinky_tofu', 'Steamed_stuffed_bun', 
            'Fried_Rice', 'Lasagne', 'Chocolate', 'Crab', 'Hamburger', 
            'Mapo_tofu', 'Hot&Sour_soup', 'Turnip_cake', 'Sushi', 
            'Bread', 'Gongbao_chicken', 'Fried_food', 'Egg_tart', 
            'Curry_chicken', 'Shrimp', 'Lobster', 'Rice_tamale', 
            'Subway', 'Ice_cream', 'Corn', 'Shumai'
        ]
        print classification_report(test_label, pred, target_names=names)
        break


    #hog = descriptor.extract_hog(dataset['path'])
    #hog = bag_of_word(hog.reshape(hog.shape[0], -1, hog.shape[-1]), dict_size=64)
    #train(label.tolist(), hog.reshape(hog.shape[0], -1).tolist(), '-v 5 -q')
