import cv2
import cv2.cv as cv
import numpy as np

from util import *

def get_color(image):
    bins = (32, 32, 32)
    ranges = [[0, 1], [0, 1], [0, 1]]
    num_block = (8, 8)

    # Compensate for exclusive upper boundary
    ranges = np.array([[pair[0], pair[1] + eps] for pair in ranges])

    block_shape = np.array(image.shape[:2]) // num_block
    hist = channel_hist(image, bins, ranges, block_shape, block_shape)

    return np.concatenate([h.reshape(-1) for h in hist])


def channel_hist(image, bins, ranges, block, step):
    num_channel = np.atleast_3d(image).shape[2]
    blocks = SlidingWindow(image.shape, block, step)
    hist_shape = [np.append(blocks.dst_shape, b) for b in bins]

    hist = [np.empty(hist_shape[c]) for c in range(num_channel)]
    for c in range(num_channel):
        for block in blocks:
            patch = image[block]
            block_hist = cv2.calcHist([patch], [c], None, [bins[c]], ranges[c])
            hist[c][block.dst] = block_hist.reshape(-1) / np.sum(block_hist)
    return hist


def color_cube(image, bins, ranges, block, step):
    num_channel = np.atleast_3d(image).shape[2]
    channels = range(num_channel)
    ranges = np.array(ranges).ravel()
    blocks = SlidingWindow(image.shape, block, step)
    hist_shape = np.append(blocks.dst_shape, bins)

    hist = np.empty(hist_shape)
    for block in blocks:
        patch = image[block]
        block_hist = cv2.calcHist([patch], channels, None, bins, ranges)
        hist[block.dst] = block_hist / np.sum(block_hist)
    return hist


if __name__ == "__main__":
    print "color_helper.py as main"
