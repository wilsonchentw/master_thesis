import cv2
import cv2.cv as cv
import numpy as np

from util import *
from hog_helper import extract_hog


def extract_phog(image, bins, level):
    # Perform Canny edge detection for shape
    gray_image = (image * 255.0).astype(np.uint8)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    mid, std = np.median(gray_image), np.std(gray_image)
    t_lo, t_hi = (mid + std * 1.0), (mid + std * 1.5)
    contour = cv2.Canny(gray_image, t_lo, t_hi, L2gradient=True) / 255.0

    # Calculate HOG of smallest cells
    image_shape = np.array(image.shape[:2])
    cell_shape = image_shape // (2 ** (level - 1))
    cell_hog = extract_hog(contour, bins, cell_shape, cell_shape)

    # Aggregate cell HOG to block HOG
    hog_shape = cell_hog.shape[:2]
    block_shape = [(2 ** idx, 2 ** idx) for idx in range(level)]
    blocks = [list(sliding_window(hog_shape, block, block))
              for block in block_shape]
    blocks = sum(blocks, [])
    phog = np.empty((len(blocks), bins))
    for idx, block in enumerate(blocks):
        block_hist = cell_hog[block].reshape(-1, bins)
        phog[idx] = np.sum(block_hist, axis=0)
    return phog


if __name__ == "__main__":
    print "phog_helper.py as main"
