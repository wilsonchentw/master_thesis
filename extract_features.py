#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import collections
import cv2.cv as cv 
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("image_list", help="list with path followed by label")
args = parser.parse_args()

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


def image_histogram(image, color="BGR", split=False):
    ColorHist = collections.namedtuple("ColorHist", "code bins ranges")

    bins = [4, 4, 4]
    ranges = [0, 256, 0, 256, 0, 256]
    hist = []

    if split:
        # Calculate channel histograms independently, then concatenate
        split_channels = cv2.split(image)
        for i, c in enumerate(split_channels):
            h = cv2.calcHist([c], [0], None, bins[i:i+1], ranges[i*2:i*2+2])
            hist.append(h)     # Concate
        return np.array(hist).reshape(-1, 1)/np.sum(hist)
    else:
        # Calculate jointly channel histogram in cubic form
        channels = range(1 if len(image.shape)==2 else image.shape[2])
        hist = cv2.calcHist([image], channels, None, bins, ranges)
        return hist.reshape(-1, 1)/np.sum(hist)


with open(args.image_list) as image_list:
    for line in image_list:
        path, label = line.strip().split(' ')
        image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

        # Normalize the image
        norm_image = normalize_image(image, 256, crop=True)

        # Calculate image histogram
        hist = image_histogram(norm_image, split=False)
        print(np.shape(hist))
