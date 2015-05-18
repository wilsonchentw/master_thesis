#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
from os import getcwd, listdir
from os.path import isfile, isdir, abspath
import sys
import random


def check_option(args):
    if len(args.f) != len(args.v):
        print "Mismatch number of arguments"
        exit(-1)
    elif len(filter(lambda x: x <= 0, args.v)) > 0:
        print("number of fold must be positive")
        exit(-2)
    else:
        fout = [open(f, mode="w") for f in args.f]
        return abspath(args.dataset), fout, args.v 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to dataset directory")
    parser.add_argument("-f", nargs="+", type=str, metavar="list_name")
    parser.add_argument("-v", nargs="+", type=int, metavar="fold")
    args = parser.parse_args()

    rootpath, fout, fold = check_option(args)
    cwdlist = [os.path.join(rootpath, d) for d in listdir(rootpath)]
    for label, dirpath in enumerate(filter(isdir, cwdlist)):
        image_list = [os.path.join(dirpath, f) for f in listdir(dirpath)]
        image_list = filter(isfile, image_list)
        random.shuffle(image_list)

        # Partition images list in directory
        total = len(image_list)
        if total < sum(fold):
            print "{0} only has {1} images, pass".format(dirpath, total)
            continue
        else:
            num_image = [v * total // sum(fold) for v in fold]
            sample = random.sample(range(len(fold)), total - sum(num_image))
            for idx, f in enumerate(fout):
                num = num_image[idx] + (1 if idx in sample else 0)
                image_sublist = image_list[:num]
                image_list = image_list[num:]
                for image in image_sublist:
                    f.write("{0} {1}\n".format(image, label))

