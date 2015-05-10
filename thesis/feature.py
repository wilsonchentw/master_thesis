import cv2
import cv2.cv as cv
import numpy as np

import descriptor
from util import *

__all__ = ["extract_all"]

def extract_HOG(image, bins, cell, block):
    hog = descriptor.oriented_grad_hist(image, bins, block, cell)
    return hog.reshape(-1)

    """
    image_shape = np.array(image.shape[:2])
    cell_shape = np.array(cell)
    block_shape = np.array(block)

    # Extract oriented gradient histogram
    hog = descriptor.oriented_grad_hist(image, bins, cell)

    # Pool and normalize in block level
    block_num = (image_shape - block_shape) // cell_shape + (1, 1)
    cell_num = block_shape // cell_shape
    cell_hist_num = np.prod(cell_num)
    hist = np.empty((np.prod(block_num), bins))
    for idx, block in enumerate(sliding_window(hog.shape, cell_num)):
        cell_hist = hog[block].reshape((cell_hist_num, bins))
        hist[idx] = np.sum(cell_hist, axis=0)
        hist[idx] /= (np.linalg.norm(hist[idx]) + eps)

    #return hist.reshape(-1)
    """

def extract_PHOG(image, bins, cell, level):
    #hog = descriptor.oriented_grad_hist(image, bins, cell)


    #gauss_pyramid = [image]
    #for idx in range(5):
    #    laplace = cv2.Laplacian(gray_image, cv2.CV_64F)
    #    gray_image = cv2.pyrDown(gray_image)
    #    gauss_pyramid.append(gray_image)
    #    imshow(laplace)
    pass


def extract_all(dataset):
    for data in dataset:
        # Normalize image size, and modify value range
        raw_image = cv2.imread(data.path, cv2.CV_LOAD_IMAGE_COLOR)
        norm_size = np.array((256, 256))
        image = normalize_image(raw_image, norm_size, crop=True) / 255.0
        image = np.sqrt(image)   # Gamma correction

        extract_HOG(image, bins=8, cell=(8, 8), block=(16, 16))
        #extract_PHOG(image, bins=8, cell=(8, 8), level=3)
