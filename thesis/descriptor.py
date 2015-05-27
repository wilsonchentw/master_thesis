import cv2
import cv2.cv as cv
import numpy as np
import scipy
import spams
from skimage.feature import local_binary_pattern

from dip import *
from util import *
from hog import get_hog
from phog import get_phog
from color import get_color
from gabor import get_gabor


def extract_descriptor(pathlist, extract, batchsize=200):
    descriptor = []
    for idx, path in enumerate(pathlist, 1):
        raw_image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
        norm_image = normalize_image(raw_image, (256, 256), crop=True)
        image = norm_image.astype(np.float32) / 255.0
        descriptor.append(extract(image))

        if (batchsize is not None) and (idx % batchsize) == 0:
            print "line {0} is done".format(idx)

    return np.array(descriptor)


def extract_hog(pathlist):
    hog = extract_descriptor(pathlist, get_hog)
    return hog.reshape(hog.shape[0], -1)


def extract_phog(pathlist):
    phog = extract_descriptor(pathlist, get_phog)
    return phog.reshape(phog.shape[0], -1)


def extract_color(pathlist):
    color = extract_descriptor(pathlist, get_color)
    return color.reshape(color.shape[0], -1)


def extract_lbp(pathlist):
    lbp = extract_descriptor(pathlist, get_lbp)
    return lbp


def get_lbp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbps = pyramid_lbp(gray_image, 1, 8, 1)
    return np.concatenate([lbp.reshape(-1) for lbp in lbps])


def pyramid_lbp(image, level, point, radius):
    bins, step = 59, np.array((radius, radius))

    lbp_image = local_binary_pattern(image, point, radius, 'nri_uniform')
    lbps = [lbp_hist(lbp_image, bins, step * 2, step)]
    for lv in range(1, level):
        image = cv2.pyrDown(image)
        lbp_image = local_binary_pattern(image, point, radius, 'nri_uniform')
        lbps.append(lbp_hist(lbp_image, bins, step * 2, step))

    return lbps


def lbp_hist(image, bins, block, step):
    blocks = SlidingWindow(image.shape, block, step)
    hist = np.zeros(np.append(blocks.dst_shape, bins))
    for block in blocks:
        patch = image[block].astype(int).reshape(-1)
        hist[block.dst] = np.bincount(patch, minlength=bins)
        hist[block.dst] = np.sqrt(hist[block.dst] / (np.sum(block.dst) + eps))
    return hist
