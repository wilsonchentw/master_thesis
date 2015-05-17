import sys

import cv2
import cv2.cv as cv
import numpy as np

from hog import extract_hog
from phog import extract_phog
from color import extract_color
from gabor import extract_gabor
from util import *

def extract_descriptor(label, path):
    data = {'label': int(label)}

    # Preprocess image
    raw_image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
    image = normalize_image(raw_image, (256, 256), crop=True)
    image = image.astype(np.float32) / 255.0
    enhance_image = get_clahe(image)
    contour = canny_edge(image)

    # Histogram of Oriented Gradient
    hog = extract_hog(image, bins=12, block=(32, 32), step=(16, 16))
    data['hog'] = hog.reshape(-1)

    # Pyramid of Histogram of Oriented Gradient
    phog = extract_phog(contour, bins=12, level=4)
    data['phog'] = np.concatenate([h.reshape(-1) for h in phog])

    # Blockwise color histogram
    bins = (32, 32, 32)
    ranges = [[0, 1], [0, 1], [0, 1]]
    param = {'num_block': (4, 4),  'bins': bins, 'ranges': ranges, }
    color = extract_color(image, **param)
    data['color'] = np.concatenate([c.reshape(-1) for c in color])

    # Gabor filter bank response
    gabor = extract_gabor(image, num_block=(4, 4), param_bank=None)
    data['gabor'] = gabor.reshape(-1)

    return data


if __name__ == "__main__":
    print "descriptor.py as main"
