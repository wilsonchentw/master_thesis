import cv2
import cv2.cv as cv
import numpy as np

from util import *
from hog import extract_hog


def extract_phog(image, bins, level):
    # Calculate HOG of smallest cells
    image_shape = np.array(image.shape[:2])
    cell_shape = image_shape // (2 ** (level - 1))
    hog = extract_hog(image, bins, cell_shape, cell_shape)

    # Summarize HOG of blocks
    block_shape = [(2 ** (level - 1 - lv),) * 2 for lv in range(level)]
    blocks = [SlidingWindow(hog.shape, blk, blk) for blk in block_shape]
    phog_shape = [np.append(blks.dst_shape, bins) for blks in blocks]
    phog = [np.empty(ps) for ps in phog_shape]
    for lv in range(level):
        level_blocks = blocks[lv]
        for block in level_blocks:
            block_hist = hog[block].reshape(-1, bins)
            phog[lv][block.dst] = np.sum(block_hist, axis=0)

    return phog


if __name__ == "__main__":
    print "phog_helper.py as main"
