import cv2
import cv2.cv as cv
import numpy as np

from util import *

def extract_hog(image, bins, block, step=None):
    image_shape = np.array(image.shape[:2])
    block_shape = np.array(block)
    block_num = (image_shape - block_shape) // step + (1, 1)
    step = (block_shape if step is None else np.array(step))

    # Compute gradient & orientation, then quantize angle int bins
    magnitude, angle = get_gradient(image)
    angle = (angle / (np.pi * 2.0) * bins).astype(int)

    # For multiple channel, choose largest gradient norm as magnitude
    magnitude, angle = map(np.atleast_3d, (magnitude, angle))
    largest_idx = magnitude.argmax(axis=2)
    x, y = np.indices(largest_idx.shape)
    magnitude = magnitude[x, y, largest_idx]
    angle = angle[x, y, largest_idx]

    # Calculate histogram of each block
    hist = np.empty((np.prod(block_num), bins))
    blocks = list(sliding_window(image_shape, block_shape, step))
    for idx, block in enumerate(blocks):
        mag = magnitude[block].reshape(-1)
        ang = angle[block].reshape(-1)
        hist[idx] = np.bincount(ang, mag, minlength=bins)
        hist[idx] /= (np.linalg.norm(hist[idx]) + eps)

    return hist.reshape(np.append(block_num, bins))


if __name__ == "__main__":
    print "hog_helper.py as main"
