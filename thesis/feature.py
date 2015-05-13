import itertools

import cv2
import cv2.cv as cv
import numpy as np

import descriptor
from util import *


#def extract_HOG(image, bins, cell, block):
#    # Block overlap to achieve normalization
#    hog = descriptor.oriented_grad_hist(image, bins, block, cell)
#
#    # Normalize block to unit in L2-norm
#    hog = hog.reshape(-1, bins)
#    for idx, block in enumerate(hog):
#        hog[idx] = cv2.normalize(block, norm_type=cv2.NORM_L2).reshape(-1)
#    return hog.reshape(-1)
#
#
#def extract_PHOG(image, bins, level):
#    # Extract HOG of cell
#    image_shape = np.array(image.shape[:2])
#    cell_shape = image_shape // (2 ** (level - 1))
#    hog = descriptor.oriented_grad_hist(image, bins, cell_shape)
#
#    # Combine cell HOG into block HOG
#    block_shape = [np.array((2 ** idx, 2 ** idx)) for idx in range(level)]
#    level_blocks = [list(sliding_window(hog.shape, block, block)) 
#                    for block in block_shape]
#    blocks = sum(level_blocks, [])
#
#    phog = np.empty((len(blocks), bins))
#    for idx, block in enumerate(blocks):
#        block_hist = hog[block].reshape(-1, bins)
#        phog[idx] = np.sum(block_hist, axis=0)
#    return phog.reshape(-1)
#
#
#def extract_color(image, num_block):
#    bins, ranges = [32, 32, 32], [[0, 1], [0, 1], [0, 1]]
#    block_shape = np.array(image.shape[:2]) // num_block
#    blocks = sliding_window(image.shape, block_shape, block_shape)
#    hists = np.empty((np.prod(num_block), np.sum(bins)))
#    for idx, block in enumerate(blocks):
#        hist = descriptor.color_hist(image, bins, ranges, split=True)
#        hists[idx] = hist.reshape(-1) / np.sum(hist)
#
#    return hists.reshape(-1)
#
#
#def extract_gabor(image, num_block):
#    ksize = (15, 15)
#    sigma = min(ksize) / 6.0
#    thetas = np.linspace(0, np.pi, num=6, endpoint=False)
#    lambds = min(ksize) / np.arange(3.0, 3.5, 0.5)
#    gammas = [0.5]
#
#    image_shape = image.shape[:2]
#    block_shape = np.array(image_shape) // num_block
#    gray_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
#    gray_image = gray_image.astype(float)
#
#    param_bank = list(itertools.product(thetas, lambds, gammas))
#    gabor = []
#    for param_idx, param in enumerate(param_bank): 
#        theta, lambd, gamma = param
#        param = {'ksize': ksize, 'sigma': sigma, 
#                 'theta': theta, 'lambd': lambd, 'gamma': gamma}
#        real, imag = descriptor.gabor_response(gray_image, **param)
#
#        # Extract blockwise mean and variance
#        magnitude = np.sqrt((real ** 2) + (imag ** 2))
#        blocks = list(sliding_window(image_shape, block_shape, block_shape))
#        for block in blocks:
#            gabor += [np.mean(magnitude[block]), np.var(magnitude[block])]
#    return np.array(gabor)


if __name__ == "__main__":
    print "feature.py as main"
