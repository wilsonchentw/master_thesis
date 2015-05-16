import sys

import cv2
import cv2.cv as cv
import numpy as np

from hog import extract_hog
from phog import extract_phog
from color import extract_color
from gabor import extract_gabor
from util import *


def extract_feature(data):
    # Preprocess image
    raw_image = cv2.imread(data.path, cv2.CV_LOAD_IMAGE_COLOR)
    image = normalize_image(raw_image, (256, 256), crop=True)
    image = image.astype(np.float32) / 255.0
    enhance_image = get_clahe(image)
    contour = canny_edge(image)

    # Histogram of Oriented Gradient
    hog = extract_hog(image, bins=12, block=(16, 16), step=(8, 8))
    data.hog = hog.reshape(-1)

    # Pyramid of Histogram of Oriented Gradient
    #phog = extract_phog(contour, bins=12, level=3)

    #color_param = {
    #    'num_block': (4, 4),  
    #    'bins': (32, 32, 32), 
    #    'ranges': [[0, 1], [0, 1], [0, 1]], 
    #    'split': True
    #}
    #data.color = extract_color(image, **color_param)
    #data.gabor = extract_gabor(image, num_block=(4, 4), param_bank=None)


if __name__ == "__main__":
    print "feature.py as main"
