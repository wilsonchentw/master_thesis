import cv2
import cv2.cv as cv
import numpy as np

from dip import *
from util import *
from hog import raw_hog


def vgg_phog(image, level, bins):
    cell_shape = np.array(image.shape[:2]) // (2 ** (level - 1))

    phog = [raw_hog(image, bins, cell_shape, cell_shape)]
    for lv in range(1, level):
        blocks = SlidingWindow(phog[lv - 1].shape, (2, 2), (2, 2))
        phog.append(np.zeros(np.append(blocks.dst_shape, bins)))
        for block in blocks:
            block_hist = phog[lv - 1][block].reshape(-1, bins)
            phog[lv][block.dst] = np.sum(block_hist, axis=0)

    return phog


def dpm_hog(image, level, bins, block, cell):
    cell_shape = np.array((16, 16))

    phog = [raw_hog(image, bins, cell_shape, cell_shape * 2)]
    for lv in range(1, level):
        image = cv2.pyrDown(image)
        phog.append(raw_hog(image, bins, block, cell))

    return phog


def get_phog(image):
    phog = vgg_phog(image, level=3, bins=32)
    #phog = dpm_hog(image, level=3, bins=32, block=(64, 64), cell=(32, 32))
    phog = np.concatenate([hog.reshape(-1) for hog in phog])
    return phog
