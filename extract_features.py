#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import collections
import itertools
import cv2.cv as cv 
import cv2
import numpy as np
import scipy

def show(image, time=0):
    cv2.imshow("image", image)
    cv2.waitKey(time)


def flatten(nested_list):
    for element in nested_list:
        is_iterable = isinstance(element, collections.Iterable)
        if is_iterable and not isinstance(element, (str, bytes)):
            for unit in flatten(element):
                    yield unit
        else:
            yield element
            

def normalize_image(image, norm_size, crop=True):
    if not crop:   
        # Directly resize the image without cropping
        return cv2.resize(image, (norm_size, norm_size))
    else:           
        # Normalize shorter side to norm_size
        height, width, channels = image.shape
        scale = max(float(norm_size)/height, float(norm_size)/width)
        norm_image = cv2.resize(src=image, dsize=(0, 0), fx=scale, fy=scale)

        # Crop for central part image
        height, width, channels = norm_image.shape
        y, x = (height-norm_size)//2, (width-norm_size)//2
        return norm_image[y:y+norm_size, x:x+norm_size]


def image_histogram(image, color=-1, split=False):
    ColorHist = collections.namedtuple("ColorHist", "bins ranges")
    color_space = {
        -1:             ColorHist([32, 32, 32], [0, 256, 0, 256, 0, 256]),
        cv.CV_BGR2GRAY: ColorHist([32],         [0, 256]                ),
        cv.CV_BGR2HSV:  ColorHist([32, 32, 32], [0, 180, 0, 256, 0, 256]),
        cv.CV_BGR2Lab:  ColorHist([32, 32, 32], [0, 256, 1, 256, 1, 256])
    }

    # Convert image to specific color space, and prepare color model   
    image = image if color== -1 else cv2.cvtColor(image, color)
    bins = color_space[color].bins
    ranges = color_space[color].ranges
    hist = []

    if split:
        # Calculate channel histograms independently, then concatenate
        split_channels = cv2.split(image)
        for i, c in enumerate(split_channels):
            h = cv2.calcHist([c], [0], None, bins[i:i+1], ranges[i*2:i*2+2])
            hist.append(h)     
        return np.array(hist)/np.sum(hist)
    else:
        # Calculate jointly channel histogram in cubic form
        channels = range(1 if len(image.shape)==2 else image.shape[2])
        hist = cv2.calcHist([image], channels, None, bins, ranges)
        return hist/np.sum(hist)


def write_in_libsvm(label, ndarray, fout):
    fout.write(label)
    for (idx,), value in np.ndenumerate(ndarray.reshape(-1)):
        if value != 0:
            fout.write(" " + str(idx+1) + ":" + str(value))
    fout.write("\n")


def sliding_window(image, window, step):
    start = np.zeros(len(window), np.int8)
    window = np.array(window)
    stop = np.array(image.shape[0:len(window)])-window+1
    grids = [np.array(range(a, b, c)) for a, b, c in zip(start, stop, step)]
    for offset in itertools.product(*grids):
        block = [slice(a, b) for a, b in zip(offset, offset+window)]
        yield image[block]


parser = argparse.ArgumentParser()
parser.add_argument("fin", metavar="image_list", 
                    help="list with path followed by label")
parser.add_argument("fout", metavar="features", 
                    help="image features in libsvm format")
args = parser.parse_args()
with open(args.fin, 'r') as fin, open(args.fout, 'w') as fout:
    for line in fin:
        path, label = line.strip().split(' ')
        image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

        # Normalize the image
        norm_size = 64
        norm_image = normalize_image(image, norm_size, crop=True)

        # Output raw image
        write_in_libsvm(label, norm_image/255.0, fout)
        """
        # Generate concatenate color histogram
        concat_hist = []
        window = (norm_size/4, norm_size/4)
        stride = (norm_size/4, norm_size/4)
        for patch in sliding_window(norm_image, window, stride):
            hist = image_histogram(patch, color=-1, split=True)
            concat_hist.append(hist)
        concat_hist = np.array(concat_hist).reshape(-1)
        write_in_libsvm(label, concat_hist, fout)
        """