#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("image_list", help="list with path followed by label")
args = parser.parse_args()

def normalize_image(image, norm_size, crop=True):
    if not crop:   
        # Directly resize the image without cropping
        return cv2.resize(image, (norm_size, norm_size))
    else:           
        # Normalize shorter side to norm_size
        height, width, channels = image.shape
        scale = max(float(norm_size)/height, float(norm_size)/width)
        norm_image = cv2.resize(src=image, dsize=(0, 0), fx=scale, fy=scale)

        # Crop for central part image
        height, width, channels = norm_image.shape
        y, x = ((height-norm_size)//2, (width-norm_size)//2)
        return norm_image[y:y+norm_size, x:x+norm_size]

with open(args.image_list) as image_list:
    for line in image_list:
        path, label = line.strip().split(' ')
        image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

        ## Normalize the image
        norm_image = normalize_image(image, 256, True)

        
