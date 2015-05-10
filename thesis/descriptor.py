import itertools

import cv2
import cv2.cv as cv
import numpy as np

from util import *

__all__ = [
    "get_gradient", "get_HOG", "color_hist", "gabor_magnitude", "extract_all", 
]

def get_gradient(image):
    # Solve gradient angle & magnitude
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y)
    angle %= (np.pi * 2)    # Truncate angle exceed 2PI

    return magnitude, angle


def get_HOG(image, bins, block, cell):
    image_shape = np.array(image.shape[:2])
    block_shape = np.array(block)
    cell_shape = np.array(cell)

    # Compute gradient & orientation, then quantize angle int bins
    magnitude, angle = get_gradient(image)
    angle = (angle / (np.pi * 2) * bins).astype(int)

    # For multiple channel, choose largest gradient norm as magnitude
    largest_idx = magnitude.argmax(axis=2)
    x, y = np.indices(largest_idx.shape)
    magnitude = magnitude[x, y, largest_idx]
    angle = angle[x, y, largest_idx]

    # Calculate block HOG directly
    block_num = (image_shape - block_shape) // cell_shape + (1, 1)
    hist = np.empty((np.prod(block_num), bins))
    blocks = sliding_window(image_shape, block_shape, cell_shape)
    for idx, block in enumerate(blocks):
        mag = magnitude[block].reshape(-1)
        ang = angle[block].reshape(-1)
        hist[idx] = np.bincount(ang, mag, minlength=bins)
        hist[idx] /= (np.linalg.norm(hist[idx]) + eps)

    return hist.reshape(-1)


def color_hist(image, color=-1, split=False):
    quantize_range = {
        -1: ([2, 2, 2], [[0, 1], [0, 1], [0, 1]]), 
        cv2.COLOR_BGR2LAB: ([2, 2, 2], [[0, 100], [-127, 127], [-127, 127]]), 
        cv2.COLOR_BGR2HSV: ([2, 2, 2], [[0, 1], [0, 1], [0, 360]]), 
    }

    image = image.astype(np.float32)
    image = (image if color == -1 else cv2.cv2Color(image, color))
    num_channels = (1 if len(image.shape) == 2 else image.shape[2])
    bins, ranges = quantize_range[color]

    hists = [];
    if split:
        for c in range(0, num_channels):
            hist = cv2.calcHist([image], [c], None, [bins[c]], ranges[c])
            hist = cv2.normalize(hist, norm_type=cv2.NORM_L1)
            hists = np.append(hists, hist)
    else:
        channels = range(0, num_channels)
        ranges = np.array(ranges)
        hists = cv2.calcHist([image], channels, None, bins, ranges.ravel())
        hists = cv2.normalize(hists, norm_type=cv2.NORM_L1)
    return np.array(hists)


def gabor_magnitude(image, kernel_size=(11, 11)):
    ksize = tuple(np.array(kernel_size))
    sigma = [min(ksize)/6.0]
    theta = np.linspace(0, np.pi, num=6, endpoint=False)
    lambd = min(ksize)/np.arange(5.0, 0.0, -1)
    gamma = [1.0]

    image = image.astype(np.float32)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    gabor_features = []
    for param in itertools.product([ksize], sigma, theta, lambd, gamma):
        ksize, sigma, theta, lambd, gamma = param
        real = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, 0)
        imag = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, np.pi/2)

        response_real = cv2.filter2D(gray_image, cv.CV_64F, real)
        response_imag = cv2.filter2D(gray_image, cv.CV_64F, imag)
        magnitude = np.sqrt(response_real**2+response_imag**2)
        gabor_features.extend([np.mean(magnitude), np.var(magnitude)])
    return np.array(gabor_features)


def extract_all(image):
    descriptor_dict = {
        'HOG': get_HOG(image, bins=8, block=(16, 16), cell=(8, 8)), 
        'color': color_hist(image, color=-1, split=True), 
        'gabor': gabor_magnitude(image, kernel_size=(3, 3)), 
    }
    return descriptor_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fin", metavar="image_list", 
                        help="list with path followed by label")

    print "descriptor.py as main"
