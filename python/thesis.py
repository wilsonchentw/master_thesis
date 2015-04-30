#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import collections
import inspect
import itertools
import operator
import os
import sys
import cv2
import cv2.cv as cv
import numpy as np
import scipy
import spams
from svmutil import *
from liblinearutil import *


def imshow(image, time=0):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    if inspect.isgenerator(image):
        for block in image:
            cv2.imshow("image", block)
            cv2.waitKey(time) & 0xFF
    else:
        cv2.imshow("image", image)
        cv2.waitKey(time) & 0xFF


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


def gabor_magnitude(image, kernel_size=(9, 9)):
    ksize = tuple(np.array(kernel_size))
    sigma = [min(ksize)/6]
    theta = np.linspace(0, np.pi, num=6, endpoint=False)
    lambd = min(ksize)/np.arange(5, 0, -1)
    gamma = [1.0]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_features = []
    for param in itertools.product([ksize], sigma, theta, lambd, gamma):
        ksize, sigma, theta, lambd, gamma = param
        real = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, 0)
        imag = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, np.pi/2)

        response_real = cv2.filter2D(gray_image, cv.CV_64F, real)
        response_imag = cv2.filter2D(gray_image, cv.CV_64F, imag)
        magnitude = np.sqrt(response_real**2+response_imag**2)
        gabor_features.extend([np.mean(magnitude), np.var(magnitude)])
    return gabor_features


def im2col(image, window, step):
    shape = image.shape[:2]
    num_channel = 1 if len(image.shape) == 2 else image.shape[2]

    window, step = np.array(window), np.array(step)
    num_block = (shape - window) // step + (1, 1)
    if all(num_block > 0):
        num_element = window[0] * window[1] * num_channel
        num_window = num_block[0] * num_block[1]

        col = np.empty([num_window, num_element], dtype=np.uint8, order='C')
        for idx, block in enumerate(sliding_window(image, window, step)):
            col[idx] = np.reshape(block, -1, order='A')
        return col


if __name__ == "__main__":

    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("fin", metavar="image_list", 
                        help="list with path followed by label")

    # Start parsing image list
    args = parser.parse_args()
    with open(args.fin, 'r') as fin:
        for line in fin:
            path, label = line.strip().split(' ')
            raw_image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

            # Normalize the image
            norm_size = np.array((256, 256))
            image = normalize_image(raw_image, norm_size, crop=True)

            
            window = (128, 128)
            step = (128, 128)
            col = im2col(image, window, step)
            for idx in range(col.shape[0]):
                imshow(np.reshape(col.T[:, idx], window + (3,)))


            # Calculate color histogram
            #color = cv2.COLOR_BGR2LAB
            #color_hist = color_histogram(image, color, split=True)

            # Gabor filter bank magnitude
            #gabor_magnitude(image, kernel_size=(16, 16))

            #imshow(sliding_window(image, window=(8, 8), step=(8, 8)), time=100)
            break

