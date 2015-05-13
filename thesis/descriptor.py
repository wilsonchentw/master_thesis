import cv2
import cv2.cv as cv
import numpy as np

from util import *

__all__ = [
    "get_gradient", "oriented_grad_hist", "color_hist", "gabor_response", 
    "detect_corner", 
]

def get_gradient(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y)

    # Truncate angle exceed 2PI or PI (if ignore sign of orientation)
    angle %= (np.pi * 2)
    return magnitude, angle


def oriented_grad_hist(image, bins, block, step=None):
    image_shape = np.array(image.shape[:2])
    block_shape = np.array(block)
    step = (block if step is None else step)

    # Compute gradient & orientation, then quantize angle int bins
    magnitude, angle = get_gradient(image)
    angle = (angle / (np.pi * 2.0) * bins).astype(int)

    # For multiple channel, choose largest gradient norm as magnitude
    magnitude, angle = map(np.atleast_3d, (magnitude, angle))
    largest_idx = magnitude.argmax(axis=2)
    x, y = np.indices(largest_idx.shape)
    magnitude = magnitude[x, y, largest_idx]
    angle = angle[x, y, largest_idx]

    block_num = (image_shape - block_shape) // step + (1, 1)
    hist = np.empty((np.prod(block_num), bins))
    cells = list(sliding_window(image_shape, block_shape, step))
    for idx, cell in enumerate(cells):
        mag = magnitude[cell].reshape(-1)
        ang = angle[cell].reshape(-1)
        hist[idx] = np.bincount(ang, mag, minlength=bins)

    return hist.reshape(np.append(block_num, bins))


def color_hist(image, bins, ranges, split=True):
    # Compensate for exclusive upper boundary
    ranges = [[pair[0], pair[1] + eps] for pair in ranges]

    image = image.astype(np.float32)
    num_channel = (1 if len(image.shape) == 2 else image.shape[2])
    if split:
        hists = []
        for c in range(num_channel):
            hist = cv2.calcHist([image], [c], None, [bins[c]], ranges[c])
            hists.append(hist.reshape(-1))
    else:
        channels = range(num_channel)
        ranges = np.array(ranges).ravel()
        hists = cv2.calcHist([image], channels, None, bins, ranges)

    return np.array(hists)


def gabor_response(image, ksize, sigma, theta, lambd, gamma):
    real = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, 0)
    imag = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, cv.CV_PI/2.0)

    response_real = cv2.filter2D(image, cv.CV_64F, real)
    response_imag = cv2.filter2D(image, cv.CV_64F, imag)
    return response_real, response_imag


if __name__ == "__main__":
    print "descriptor.py as main"
