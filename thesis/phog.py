import cv2
import cv2.cv as cv
import numpy as np

from dip import *
from util import *
from hog import raw_hog


def raw_phog(image, bins, level):
    # Extract HOG of smallest cells
    image_shape = np.array(image.shape[:2])
    cell_shape = image_shape // (2 ** level)
    hog = raw_hog(image, bins, cell_shape, cell_shape)

    # Summarize HOG of blocks
    block_shape = [(2 ** lv, 2 ** lv) for lv in reversed(range(1, level + 1))]
    block_shape = [np.array(block) for block in block_shape]
    blocks = [SlidingWindow(hog.shape, blk, blk) for blk in block_shape]
    phog_shape = [np.append(blks.dst_shape, bins) for blks in blocks]
    phog = [np.empty(ps) for ps in phog_shape]
    for lv in range(level):
        for block in blocks[lv]:
            block_hist = hog[block].reshape(-1, bins)
            phog[lv][block.dst] = np.sum(block_hist, axis=0)
            #phog[lv][block.dst] /= (np.sum(phog[lv][block.dst]) + eps)
            phog[lv][block.dst] /= (np.linalg.norm(phog[lv][block.dst]) + eps)

    return phog


def vgg_phog(image, bins, level):
    cell_shape = np.array(image.shape[:2]) // (2 ** (level - 1))

    phog = [soft_hog(image, bins, cell_shape, cell_shape)]
    for lv in range(1, level):
        blocks = SlidingWindow(phog[lv - 1].shape, (2, 2), (2, 2))
        phog.append(np.zeros(np.append(blocks.dst_shape, bins)))
        for block in blocks:
            block_hist = phog[lv - 1][block].reshape(-1, bins)
            phog[lv][block.dst] = np.sum(block_hist, axis=0)

    return phog


def soft_hog(image, bins, block, step):
    # Compute gradient & orientation
    magnitude, angle = map(np.atleast_3d, get_gradient(image))

    # For multiple channel, choose largest gradient norm as magnitude
    largest_idx = magnitude.argmax(axis=2)
    x, y = np.indices(largest_idx.shape)
    magnitude = magnitude[x, y, largest_idx]
    angle = angle[x, y, largest_idx]

    # Soft assignment with linear distributed
    unit_angle = (np.pi * 2) / bins
    angle = (angle / (np.pi * 2.0) * bins).astype(int)
    weight = 1.0 - np.fmod(angle, unit_angle) / unit_angle
    
    # For each sliding window, compute histogram
    blocks = SlidingWindow(image.shape, block, step)
    hog = np.zeros(np.append(blocks.dst_shape, bins))
    for block in blocks:
        lo_bin = angle[block].reshape(-1)
        hi_bin = np.mod(lo_bin + 1, bins)
        lo_mag = (magnitude[block] * weight[block]).reshape(-1)
        hi_mag = magnitude[block].reshape(-1) - lo_mag
        lo_part = np.bincount(lo_bin, lo_mag, minlength=bins)
        hi_part = np.bincount(hi_bin, hi_mag, minlength=bins)
        hog[block.dst] = lo_part + hi_part

    return hog


def get_phog(image):
    #image = canny_edge(image)
    phog = vgg_phog(image, bins=16, level=3)
    phog = np.concatenate([hog.reshape(-1) for hog in phog])
    return phog
