#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import collections
import inspect
import itertools
import math
import operator
import os
import sys
import time
import warnings
import cv2
import cv2.cv as cv
import numpy as np
import scipy
import sklearn
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


def sliding_window(shape, window, step):
    start = np.zeros(len(window), np.int32)
    stop = np.array(shape[:len(window)]) - window + 1
    grids = [range(a, b, c) for a, b, c in zip(start, stop, step)]
    for offset in itertools.product(*grids):
        offset = np.array(offset)
        block = [slice(a, b) for a, b in zip(offset, offset+window)]
        yield block


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


def im2row(image, window, step):
    (shape, window, step) = map(np.array, (image.shape[:2], window, step))
    num_channel = 1 if len(image.shape) == 2 else image.shape[2]

    num_window = (shape - window) // step + np.array((1, 1))
    if all(num_window > 0):
        num_row = np.prod(num_window)
        dim = np.prod(window) * num_channel
        row = np.empty((num_row, dim), order='C')
        for idx, block in enumerate(sliding_window(shape, window, step)):
            row[idx] = image[block].reshape(-1, order='C')
        return row


def row2im(row, shape, window, step):
    if len(row.shape) == 1: 
        return row.reshape(tuple(shape) + (-1,))

    shape, window, step = map(np.array, (shape, window, step))
    num_channel = row.shape[1] // (window[0] * window[1])
    image = np.zeros(np.append(shape, num_channel), order='C')
    for idx, block in enumerate(sliding_window(shape, window, step)):
        image[block] = row[idx].reshape(np.append(window, num_channel))
    return image


def basis_image(basis, window):
    basis_min = np.amin(basis, 0)
    basis_max = np.amax(basis, 0)
    norm_basis = (basis - basis_min)/(basis_max - basis_min + 1e-12)

    width = math.ceil(math.sqrt(basis.shape[1]))
    height = math.ceil(basis.shape[1] / width)
    shape = np.array(window) * np.array((height, width))

    return row2im(norm_basis.T, shape, window, window)


def histogram_of_gradient(image, num_bin, window, cell):
    window, cell = np.array(window), np.array(cell)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y)

    # Choose largest gradient norm as magnitude
    largest_idx = magnitude.argmax(axis=2)
    x, y = np.indices(largest_idx.shape)
    magnitude = magnitude[x, y, largest_idx]
    angle = angle[x, y, largest_idx]
    
    image_shape = np.array(image.shape[:2])
    num_block = (image_shape - window) // cell + np.array((1, 1))
    hist = np.empty((np.prod(num_block), num_bin))
    for idx, block in enumerate(sliding_window(image_shape, window, cell)):
        mag, ang = magnitude[block].reshape(-1), angle[block].reshape(-1)
        hist[idx], edge = np.histogram(ang, num_bin, (0, np.pi*2), weights=mag)
        hist[idx] = hist[idx] / (np.linalg.norm(hist[idx]) + 1e-15)
    return hist.reshape(-1)


class Image(object):
    """ Store extracted features and normalized image """
    norm_size = np.array((256, 256))

    def __init__(self, path, label):
        self.path = path
        self.label = label

        # Normalize the image
        raw_image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
        self.image = normalize_image(raw_image, self.norm_size, crop=True)


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("fin", metavar="image_list", 
                        help="list with path followed by label")

    # Parsing image list & extract feature
    args = parser.parse_args()
    with open(args.fin, 'r') as fin:
        dataset = []
        for line in fin:
            path, label = line.strip().split(' ')
            data = Image(path, label)

            image = np.copy(data.image)/255.0
            gray_image = cv2.cvtColor(data.image, cv2.COLOR_BGR2GRAY)/255.0

            num_bin = 8
            window = (32, 32)
            cell = (16, 16)
            hog = histogram_of_gradient(image, num_bin, window, cell)
 
            #gauss_pyramid = [image]
            #for idx in range(5):
            #    laplace = cv2.Laplacian(gray_image, cv2.CV_64F)
            #    gray_image = cv2.pyrDown(gray_image)
            #    gauss_pyramid.append(gray_image)
            #    imshow(laplace)
