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


def gabor_param():
    # Default parameter is according to Serre [PAMI'06] except ksize
    ksizes = range(7, 39, 4)
    sigmas = []
    thetas = np.linspace(0, np.pi, num=4, endpoint=False)
    lambds = []
    gammas = []

    param_bank = []
    for theta, ksize in itertools.product(thetas, ksizes):
        sigma = 0.0036 * (ksize ** 2) + 0.35 * ksize + 0.18
        param = {
            'ksize': (ksize, ksize), 
            'sigma': sigma, 
            'theta': theta, 
            'lambd': sigma / 0.8, 
            'gamma': 0.3, 
        }
        param_bank.append(param)
    return param_bank


def extract_gabor(image, num_block, param_bank=None):
    # Setup Gabor filter bank parameter
    param_bank = gabor_param() if param_bank is None else param_bank
    num_param = len(param_bank)

    image_shape = image.shape[:2]
    block_shape = np.array(image_shape) // num_block
    blocks = SlidingWindow(image_shape, block_shape, block_shape)
    num_block = np.prod(blocks.dst_shape)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor = np.empty((num_param, num_block, 2))
    for param_idx, param in enumerate(param_bank):
        real, imag = gabor_response(gray_image, **param)
        magnitude = np.sqrt((real ** 2) + (imag ** 2))

        for block_idx, block in enumerate(blocks):
            patch = magnitude[block]
            gabor[param_idx, block_idx] = [np.mean(patch), np.var(patch)]
    return gabor
 

if __name__ == "__main__":
    print "gabor_helper.py as main"
