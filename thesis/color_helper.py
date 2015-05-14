import cv2
import cv2.cv as cv
import numpy as np

from util import *

def extract_color(image, num_block, bins, ranges, split):
    # Convert type for cv2.calcHist
    image = image.astype(np.float32)

    # Compensate for exclusive upper boundary
    ranges = [[pair[0], pair[1] + eps] for pair in ranges]

    block_shape = np.array(image.shape[:2]) // num_block
    blocks = list(sliding_window(image.shape, block_shape, block_shape))
    hist = np.empty((len(blocks), np.sum(bins)))
    for idx, block in enumerate(blocks):
        if split:
            split_hist = channel_hist(image[block], bins, ranges)
            hist[idx] = np.concatenate(split_hist)
        else:
            hist[idx] = color_cube(image, bins, ranges).reshape(-1)
    return hist


def channel_hist(image, bins, ranges):
    num_channel = np.atleast_3d(image).shape[2]
    hists = [np.empty(num_bin) for num_bin in bins]
    for c in range(num_channel):
        hist = cv2.calcHist([image], [c], None, [bins[c]], ranges[c])
        hists[c] = hist.reshape(-1)
    return hists


def color_cube(image, bins, ranges):
    num_channel = np.atleast_3d(image).shape[2]
    channels = range(num_channel)
    ranges = np.array(ranges).ravel()
    hist = cv2.calcHist([image], channels, None, bins, ranges)
    return hist


if __name__ == "__main__":
    print "color_helper.py as main"
