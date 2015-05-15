import itertools

import cv2
import cv2.cv as cv
import numpy as np

from util import *


def gabor_response(image, ksize, sigma, theta, lambd, gamma):
    real = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, 0)
    imag = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, cv.CV_PI/2.0)

    response_real = cv2.filter2D(image, cv.CV_32F, real)
    response_imag = cv2.filter2D(image, cv.CV_32F, imag)

    return response_real, response_imag


def set_parameter(param_bank):
    if param_bank is None:
        ksizes = [(15, 15)]
        sigmas = [min(ks) / 6.0 for ks in ksize]
        thetas = np.linspace(0, np.pi, num=6, endpoint=False)
        lambds = [sigma / 0.8 for sigma in sigmas]
        gammas = [0.3]
    else:
        ksizes = param_bank['ksize']
        sigmas = param_bank['sigma']
        thetas = param_bank['theta']
        lambds = param_bank['lambd']
        gammas = param_bank['gamma']

    # Iterate all parameter according to Serre [PAMI 06]
    param_bank = []
    for theta, gamma in itertools.product(thetas, gammas):
        for idx, ksize in enumerate(ksizes):
            param = (ksize, sigmas[idx], theta, lambds[idx], gamma)
            param_bank.append(param)
    return param_bank


def extract_gabor(image, num_block, param_bank):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_shape = image.shape[:2]
    block_shape = np.array(image_shape) // num_block
    blocks = SlidingWindow(image_shape, block_shape, block_shape)
    block_num = np.prod(blocks.dst_shape)

    # For each parameter, convolve gabor filter using given parameter
    param_bank = set_parameter(param_bank)
    gabor = np.empty((len(param_bank), block_num, 2))
    for param_idx, param in enumerate(param_bank):
        ksize, sigma, theta, lambd, gamma = param
        real, imag = gabor_response(gray, ksize, sigma, theta, lambd, gamma)

        # Extract gabor magnitude of mean and variance for each patch
        magnitude = np.sqrt((real ** 2) + (imag ** 2))
        #imshow(cv2.normalize(magnitude, norm_type=cv2.NORM_MINMAX), time=1)

        for block_idx, block in enumerate(blocks):
            patch = magnitude[block]
            gabor[param_idx, block_idx] = [np.mean(patch), np.var(patch)]
            
    return gabor


if __name__ == "__main__":
    print "gabor_helper.py as main"
