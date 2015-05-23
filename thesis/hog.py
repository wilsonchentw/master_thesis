import cv2
import cv2.cv as cv
import numpy as np


from dip import *
from util import *


def raw_hog(image, bins, block, step):
    # Compute gradient & orientation, then quantize angle int bins
    magnitude, angle = get_gradient(image)
    angle = (angle / (np.pi * 2.0) * bins).astype(int)

    # For multiple channel, choose largest gradient norm as magnitude
    magnitude, angle = map(np.atleast_3d, (magnitude, angle))
    largest_idx = magnitude.argmax(axis=2)
    x, y = np.indices(largest_idx.shape)
    magnitude = magnitude[x, y, largest_idx]
    angle = angle[x, y, largest_idx]

    # Gaussian window, downweight pixel near edge
    ksize = tuple((x + 1 if x % 2 == 0 else x) for x in block)
    sigmaX, sigmaY = block[1] * 0.5, block[0] * 0.5
    gauss_param = {'ksize': ksize, 'sigmaX': sigmaX, 'sigmaY': sigmaY}

    # Calculate weighted histogram with L2-normalization
    blocks = SlidingWindow(image.shape, block, step)
    hist_shape = np.append(blocks.dst_shape, bins)
    hist = np.empty(hist_shape)
    for block in blocks:
        mag = cv2.GaussianBlur(magnitude[block], **gauss_param).reshape(-1)
        ang = angle[block].reshape(-1)
        hist[block.dst] = np.bincount(ang, mag, minlength=bins)
        hist[block.dst] /= (np.linalg.norm(hist[block.dst]) + eps)

    return hist


def get_hog(image):
    #image = get_clahe(image)
    hog = raw_hog(image, bins=72, block=(128, 128), step=(128, 128))
    #hog = np.sqrt(hog / np.sum(hog))    # RootHOG
    return hog.reshape(-1)
