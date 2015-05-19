import sys

import cv2
import cv2.cv as cv
import numpy as np

from hog import extract_hog
from phog import extract_phog
from color import extract_color
from gabor import extract_gabor
from util import *

def instance_descriptor(label, path):
    data = {'label': int(label)}

    # Preprocessing image
    raw_image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
    image = normalize_image(raw_image, (256, 256), crop=True)
    image = image.astype(np.float32) / 255.0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhance_image = get_clahe(image)
    contour = canny_edge(image)

    data['hog'] = extract_hog(image)
    data['phog'] = extract_phog(contour)
    data['color'] = extract_color(image)
    data['gabor'] = extract_gabor(gray_image)

    return data


if __name__ == "__main__":
    print "descriptor.py as main"
