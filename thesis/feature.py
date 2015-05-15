import cv2
import cv2.cv as cv
import numpy as np

from hog_helper import extract_hog
from phog_helper import extract_phog
from color_helper import extract_color
from gabor_helper import extract_gabor
from util import *


def extract_all(data):
    # Setup parameter for descriptor
    hog_param = {'bins': 12, 'block': (16, 16), 'step': (8, 8), }
    phog_param = {'bins': 12, 'level': 3, }
    color_param = {
        'num_block': (4, 4), 
        'bins': (32, 32, 16), 
        'ranges': [[0, 1], [0, 1], [0, 1]], 
        'split': True, 
    }
    ksize = range(7, 39, 4)
    gabor_param = {
        'num_block': (4, 4), 
        'param_bank': {
            'ksize': [(ks, ks) for ks in ksize], 
            'sigma': [0.0036 * ks * ks + 0.35 * ks + 0.18 for ks in ksize], 
            'theta': np.linspace(0, np.pi, num=4, endpoint=False), 
            'lambd': [0.0045 * ks * ks + 0.4375 * ks + 0.225 for ks in ksize], 
            'gamma': [0.3], 
        }
    }

    # Preprocessing input image
    raw_image = cv2.imread(data.path, cv2.CV_LOAD_IMAGE_COLOR)
    image = normalize_image(raw_image, (256, 256), crop=True)
    image = image.astype(np.float32) / 255.0
    enhance_image = get_clahe(image)

    data.hog = extract_hog(image, **hog_param)
    data.phog = extract_phog(image, **phog_param)
    data.color = extract_color(image, **color_param)
    data.gabor = extract_gabor(enhance_image, **gabor_param)


if __name__ == "__main__":
    print "feature.py as main"
