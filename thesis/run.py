#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import warnings

import cv2
import cv2.cv as cv
import numpy as np

import feature
from util import *


class Image(object):
    """ Store path, label, and extracted descriptors """
    def __init__(self, path, label):
        self.path = path
        self.label = label


def get_CLAHE(image):
    # Global enhance luminance
    enhance_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2LAB)
    luminance = enhance_image[:, :, 0] / 100.0
    luminance = cv2.normalize(luminance, norm_type=cv2.NORM_MINMAX)
    luminance = (luminance * 255).astype(np.uint8)

    # Perform CLAHE on L-channel
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    enhance_image[:, :, 0] = clahe.apply(luminance) / 255.0 * 100.0
    enhance_image = cv2.cvtColor(enhance_image, cv2.COLOR_LAB2BGR)

    # Convert type to float
    return enhance_image.astype(float)


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

            # Normalize image size
            norm_size = np.array((256, 256))
            raw_image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
            image = normalize_image(raw_image, norm_size, crop=True) / 255.0
            enhance_image = get_CLAHE(image)

            # Extract feature
            cell, block = (16, 16), (32, 32)
            data = Image(path, label)
            data.hog = feature.extract_HOG(image, 12, cell, block)
            data.phog = feature.extract_PHOG(image, bins=12, level=4)
            data.color = feature.extract_color(image, num_block=(4, 4))
            data.gabor = feature.extract_gabor(image, num_block=(4, 4))
            dataset.append(data)

            # Progress report
            if line_idx % 10 == 0: 
                print "line{0} is done".format(line_idx)


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
