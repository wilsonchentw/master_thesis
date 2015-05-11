import cv2
import cv2.cv as cv
import numpy as np

import descriptor
from util import *

__all__ = [
    "extract_all", "extract_HOG", "extract_PHOG", "extract_color", 
    "extract_gabor", 
]

def extract_HOG(image, bins, cell, block):
    hog = descriptor.oriented_grad_hist(image, bins, block, cell)
    return hog.reshape(-1)


def extract_PHOG(image, bins, level):
    # Extract HOG of cell then combine into HOG of block
    image_shape = np.array(image.shape[:2])
    cell_shape = image_shape // (2 ** (level - 1))
    hog = descriptor.oriented_grad_hist(image, bins, cell_shape, cell_shape)
    blocks = []
    for idx in range(level):
        block_shape = np.array((1, 1)) * (2 ** idx)
        blocks += list(sliding_window(hog.shape, block_shape, block_shape))

    phog = np.empty((len(blocks), bins))
    for idx, block in enumerate(blocks):
        block_hist = hog[block].reshape(-1, bins)
        phog[idx] = np.sum(block_hist, axis=0)
    return phog.reshape(-1)


def extract_color(image, color, split):
    color_hist = descriptor.color_hist(image, color, split)
    return color_hist


def extract_gabor(image, ksize):
    gabor = descriptor.gabor_magnitude(image, ksize)
    return gabor


def extract_all(dataset):
    for data in dataset:
        # Normalize image size
        norm_size = np.array((256, 256))
        raw_image = cv2.imread(data.path, cv2.CV_LOAD_IMAGE_COLOR)
        image = normalize_image(raw_image, norm_size, crop=True)

        # Perform contrast limited adaptive histogram equalization on L-channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        contrast_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        luminance = contrast_image[:, :, 0] / 255.0
        luminance = cv2.normalize(luminance, norm_type=cv2.NORM_MINMAX) * 255.0
        contrast_image[:, :, 0] = clahe.apply(luminance.astype(np.uint8))
        contrast_image = cv2.cvtColor(contrast_image, cv2.COLOR_LAB2BGR)
        image = contrast_image / 255.0

        extract_HOG(image, bins=8, cell=(8, 8), block=(16, 16))
        extract_PHOG(image, bins=8, level=3)
        extract_color(image, color=-1, split=True)
        extract_gabor(image, ksize=(11, 11))

        #gauss_pyramid = [image]
        #for idx in range(5):
        #    laplace = cv2.Laplacian(gray_image, cv2.CV_64F)
        #    gray_image = cv2.pyrDown(gray_image)
        #    gauss_pyramid.append(gray_image)
        #    imshow(laplace)

