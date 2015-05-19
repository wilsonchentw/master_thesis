import os

import cv2
import cv2.cv as cv
import numpy as np

import dip
from hog import get_hog
from phog import get_phog
from color import get_color
from gabor import get_gabor
from utility import *

def extract_descriptor(filename, preprocess, extract, batchsize=None):
    preprocess = (lambda x: x) if preprocess is None else preprocess
    normalize_image = dip.normalize_image

    descriptor = []
    with open(filename, 'r') as fin:
        for idx, line in enumerate(fin, 1):
            path, label = line.strip().split(" ")

            raw_image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
            norm_image = normalize_image(raw_image, (256, 256), crop=True)
            image = preprocess(norm_image.astype(np.float32) / 255.0)
            descriptor.append(extract(image))

            if (batchsize is not None) and (idx % batchsize) == 0:
                print "line {0} is done".format(idx)

        return np.array(descriptor)


def extract_all(filename, batchsize=None):
    param = {
        'hog': (None, get_hog), 
        'phog': (None, get_phog), 
        'color': (None, get_color), 
        'gabor': (lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), get_gabor), 
    }

    prefix = os.path.basename(filename).partition('.')[0]
    with open(filename, 'r') as fin:
        label = [line.strip().split(" ")[1] for line in fin]

    data = {}
    for name, (preproc, extract) in param.items():
        data[name] = extract_descriptor(filename, preproc, extract, batchsize)
        datname = "{0}_{1}.dat".format(prefix, name)
        svm_write_problem(datname, label, data[name])
        
    return data


if __name__ == "__main__":
    print "descriptor.py as main"
