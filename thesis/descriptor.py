import cv2
import cv2.cv as cv
import numpy as np
import scipy
import spams

from dip import *
from util import *
from hog import raw_hog
from phog import vgg_phog, dpm_phog
from color import channel_hist, color_cube
from gabor import get_gabor


def extract_descriptor(pathlist, extract, batchsize=None):
    descriptor = []
    for idx, path in enumerate(pathlist, 1):
        raw_image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
        norm_image = normalize_image(raw_image, (256, 256), crop=True)
        image = norm_image.astype(np.float32) / 255.0
        descriptor.append(extract(image))

        if (batchsize is not None) and (idx % batchsize) == 0:
            print "line {0} is done".format(idx)

    return np.array(descriptor)


def get_hog(image):
    hog = raw_hog(image, bins=16, block=(32, 32), step=(32, 32))
    #hog = raw_hog(image, bins=128, block=(64, 64), step=(32, 32))
    #hog /= np.linalg.norm(hog.reshape(-1))
    return hog


def get_phog(image):
    phog = vgg_phog(image, level=3, bins=32)
    #phog = dpm_hog(image, level=3, bins=128, block=(64, 64), cell=(32, 32))
    phog = np.concatenate([hog.reshape(-1) for hog in phog])
    return phog


def get_color(image):
    bins = (32, 32, 32)
    ranges = [[0, 1], [0, 1], [0, 1]]
    num_block = (16, 16)

    # Compensate for exclusive upper boundary
    ranges = np.array([[pair[0], pair[1] + eps] for pair in ranges])

    block_shape = np.array(image.shape[:2]) // num_block
    hist = channel_hist(image, bins, ranges, block_shape, block_shape)

    return np.concatenate([h.reshape(-1) for h in hist])


def extract_hog(pathlist):
    hog = extract_descriptor(pathlist, get_hog)
    return hog


def extract_phog(pathlist):
    phog = extract_descriptor(pathlist, get_phog)
    return phog.reshape(phog.shape[0], -1)


def extract_color(pathlist):
    color = extract_descriptor(pathlist, get_color)
    return color.reshape(color.shape[0], -1)
