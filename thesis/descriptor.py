import os
import sys

import cv2
import cv2.cv as cv
import numpy as np

from dip import *
from util import *
from hog import get_hog
from phog import get_phog
from color import get_color
from gabor import get_gabor

def extract_descriptor(pathlist, extract, batchsize=100):
    descriptor = []
    for idx, path in enumerate(pathlist, 1):
        raw_image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
        norm_image = normalize_image(raw_image, (256, 256), crop=True)
        image = norm_image.astype(np.float32) / 255.0
        descriptor.append(extract(image))

        if (batchsize is not None) and (idx % batchsize) == 0:
            print "line {0} is done".format(idx)

    return np.array(descriptor)


def extract_all(filename):
    extract_list = ['hog', 'phog', 'color', 'gabor']

    prefix = os.path.basename(filename).partition('.')[0]
    dataset = preload_list(filename)
    for name in extract_list:
        print "Extract {0}... ".format(name)

        extract = globals()["get_" + name]
        dataset[name] = extract_descriptor(dataset['path'], extract)

        outfile = "{0}_{1}.dat".format(prefix, name)
        svm_write_problem(outfile, dataset['label'], dataset[name])

    dataset.pop('path', None)
    return dataset
