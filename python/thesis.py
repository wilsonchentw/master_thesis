#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import itertools
import collections
import scipy
import numpy as np
import cv2
import cv2.cv as cv

def imshow(image, time=0):
    cv2.imshow("image", image)
    cv2.waitKey(time)


def normalize_image(image, norm_size, crop=True):
    if not crop:   
        # Directly resize the image without cropping
        return cv2.resize(image, norm_size)
    else:           
        # Normalize shorter side to norm_size
        height, width, channels = image.shape
        norm_height, norm_width = norm_size
        scale = max(float(norm_height)/height, float(norm_width)/width)

        norm_image = cv2.resize(src=image, dsize=(0, 0), fx=scale, fy=scale)

        # Crop for central part image
        height, width, channels = norm_image.shape
        y, x = (height-norm_height)//2, (width-norm_width)//2
        return norm_image[y:y+norm_height, x:x+norm_width]


def sliding_window(image, window, step):
    start = np.zeros(len(window), np.int8)
    window = np.array(window)
    stop = np.array(image.shape[0:len(window)])-window+1
    grids = [np.array(range(a, b, c)) for a, b, c in zip(start, stop, step)]
    for offset in itertools.product(*grids):
        block = [slice(a, b) for a, b in zip(offset, offset+window)]
        yield image[block]


def color_histogram(image, color=-1, split=False):
    color_space = { 
        -1               : ([2, 2, 2], [[0, 256], [0, 256], [0, 256]]),
        cv2.COLOR_BGR2LAB: ([2, 2, 2], [[0, 256], [1, 256], [1, 256]]),
        cv2.COLOR_BGR2HSV: ([2, 2, 2], [[0, 256], [0, 256], [0, 180]]),
    }
    image = (image if color == -1 else cv2.cvtColor(image, color))
    num_channels = 1 if len(image.shape)==2 else image.shape[2]
    (bins, ranges) = color_space[color]

    hists = [];
    if split:
        for c in range(0, num_channels):
            hist = cv2.calcHist([image], [c], None, [bins[c]], ranges[c])
            hist = cv2.normalize(hist, norm_type=cv2.NORM_L1)
            hists = np.append(hists, hist)
    else:
        channels = range(0, num_channels)
        ranges = np.array(ranges).flatten()
        hists = cv2.calcHist([image], channels, None, bins, ranges)
        hists = cv2.normalize(hists, norm_type=cv2.NORM_L1)
    return hists


def gabor_magnitude(image, kernel_size=(16, 16)):
    ksize = tuple(np.array(kernel_size))
    sigma = [min(ksize)/6*12]
    theta = np.linspace(0, np.pi, num=6, endpoint=False)
    lambd = min(ksize)/np.arange(5, 0, -1)
    gamma = [1.0]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_features = []
    for param in itertools.product([ksize], sigma, theta, lambd, gamma):
        ksize, sigma, theta, lambd, gamma = param
        real = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, 0)
        imag = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, np.pi/2)

        response_r = cv2.filter2D(gray_image, cv.CV_64F, real)
        response_i = cv2.filter2D(gray_image, cv.CV_64F, imag)
        magnitude = np.sqrt(response_r**2+response_i**2)
        gabor_features.extend([np.mean(magnitude), np.var(magnitude)])
    return gabor_features
        

if __name__ == "__main__":
   # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("fin", metavar="image_list", 
                        help="list with path followed by label")
    args = parser.parse_args()

    with open(args.fin, 'r') as fin:
        for line in fin:
            path, label = line.strip().split(' ')
            raw_image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

            # Normalize the image
            norm_size = (320, 320)
            image = normalize_image(raw_image, norm_size, crop=True)
            imshow(image)

            # Calculate color histogram
            #color = cv2.COLOR_BGR2LAB
            #color_hist = color_histogram(image, color, split=True)

            # Gabor filter bank magnitude
            #gabor_magnitude(image, kernel_size=(16, 16))

            break

