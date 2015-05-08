#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import warnings

import cv2
import cv2.cv as cv
import numpy as np

import descriptor
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
        for line in fin:
            path, label = line.strip().split(' ')
            data = Image(path, int(label))
            dataset.append(data)

    # Extract descriptor from image
    for data in dataset:
        raw_image = cv2.imread(data.path, cv2.CV_LOAD_IMAGE_COLOR)

        # Normalize image size, and modify value range
        norm_size = np.array((256, 256))
        image = (normalize_image(raw_image, norm_size, crop=True)/255.0)
        image = np.sqrt(image)   # Gamma correction

        descriptor.extract_all(image)


    """
    # Do stratified K-fold validation
    labels = [data.label for data in dataset]
    folds = StratifiedKFold(labels, n_folds=5, shuffle=True)
    for train_idx, test_idx in folds:
        train_data = [dataset[idx] for idx in train_idx]
        test_data = [dataset[idx] for idx in test_idx]

        train_label = [data.label for data in train_data]
        test_label = [data.label for data in test_data]
        train_hog = [data.hog.tolist() for data in train_data]
        test_hog = [data.hog.tolist() for data in test_data]

        model = train(train_label, train_hog, "-q")
        guess, acc, val = predict(test_label, test_hog, model, "")
    """
