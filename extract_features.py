#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("image_list", help="list w/ image path followed by label")
args = parser.parse_args()

with open(args.image_list) as f:
    for line in f:
        path, label = line.strip().split(' ')
        image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

        
