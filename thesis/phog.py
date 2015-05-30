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


def dpm_phog(image, level, bins, block, cell):
    phog = [raw_hog(image, bins, block, cell)]
    for lv in range(1, level):
        image = cv2.pyrDown(image)
        hog = raw_hog(image, bins, block, cell)
        phog.append(hog / np.linalg.norm(hog.reshape(-1)))

    return phog
