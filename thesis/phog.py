import cv2
import cv2.cv as cv
import numpy as np

from util import *
from hog import raw_hog

def raw_phog(image, bins, level):
    # Extract HOG of smallest cells
    image_shape = np.array(image.shape[:2])
    cell_shape = image_shape // (2 ** (level - 1))
    hog = raw_hog(image, bins, cell_shape, cell_shape)

    # Summarize HOG of blocks
    block_shape = [(2 ** lv, 2 ** lv) for lv in reversed(range(level))]
    blocks = [SlidingWindow(hog.shape, blk, blk) for blk in block_shape]
    phog_shape = [np.append(blks.dst_shape, bins) for blks in blocks]
    phog = [np.empty(ps) for ps in phog_shape]
    for lv in range(level):
        for block in blocks[lv]:
            block_hist = hog[block].reshape(-1, bins)
            phog[lv][block.dst] = np.sum(block_hist, axis=0)

    return phog


def extract_phog(image):
    phog = raw_phog(image, bins=12, level=4)
    return np.concatenate([hog.reshape(-1) for hog in phog])


if __name__ == "__main__":
    print "phog_helper.py as main"
