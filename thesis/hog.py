import cv2
import cv2.cv as cv
import numpy as np

from dip import *
from util import *


def raw_hog(image, bins, block, step, soft=False):
    # Compute gradient & orientation
    magnitude, angle = map(np.atleast_3d, get_gradient(image))

    # For multiple channel, choose largest gradient norm as magnitude
    largest = magnitude.argmax(axis=2)
    x, y = np.indices(largest.shape)
    magnitude, angle = map(lambda arr: arr[x, y, largest], (magnitude, angle))

    # TODO: BILINEAR INTERPOLATION WITH SPATIAL & ORIENTATION
    # Soft assignment with linear distributed
    unit_angle = (np.pi * 2) / bins
    angle = (angle / (np.pi * 2.0) * bins).astype(int)
    weight = (
        1.0 - (np.fmod(angle, unit_angle) / unit_angle) 
        if soft 
        else np.ones_like(angle)
    )

    # Prepare Gaussian window for downweight pixel near edge
    gauss_param = {
        'ksize': tuple((x + 1 if x % 2 == 0 else x) for x in block), 
        'sigmaX': block[1] * 0.5, 
        'sigmaY': block[0] * 0.5, 
    }

    # For each sliding window, compute histogram
    blocks = SlidingWindow(image.shape, block, step)
    hog = np.zeros(np.append(blocks.dst_shape, bins))
    for block in blocks:
        lo_bin = angle[block].reshape(-1)
        hi_bin = np.mod(lo_bin + 1, bins)

        mag = cv2.GaussianBlur(magnitude[block], **gauss_param)
        lo_mag = (mag * weight[block]).reshape(-1)
        hi_mag = mag.reshape(-1) - lo_mag

        lo_part = np.bincount(lo_bin, lo_mag, minlength=bins)
        hi_part = np.bincount(hi_bin, hi_mag, minlength=bins)
        hog[block.dst] = lo_part + hi_part

        # Using Bhattacharya distance (L1-sqrt)
        hog[block.dst] /= (np.sum(hog[block.dst]) + eps)
        hog[block.dst] = np.sqrt(hog[block.dst])

    return hog


def get_hog(image):
    hog = raw_hog(image, bins=128, block=(64, 64), step=(32, 32))
    hog /= np.linalg.norm(hog.reshape(-1))
    return hog.reshape(-1)
