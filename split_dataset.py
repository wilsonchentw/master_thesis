#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import random
import argparse
from itertools import accumulate
from os import getcwd, listdir
from os.path import isfile, join, abspath

def check_option(args):
    if not args.f and not args.v:
        args.f = [sys.stdout]
        args.v = [1]
    elif len(args.f) != len(args.v):
        print("mismatch number of arguments")
        exit(1)
    elif len([e for e in args.v if e < 0]) > 0:
        print("number of fold must be non-negative")
        exit(1)
    else:
        args.f = [open(f, mode="w") for f in args.f]
    return abspath(args.dataset), args.f, args.v 

def usage_msg():
    return ("split_dataset.py dataset_folder [-f list ...] [-v portion ...]\n"
            "If no -f & -v, print all files in dataset_folder on screen\n "
           )

parser = argparse.ArgumentParser(usage=usage_msg())
parser.add_argument("dataset", help="path to dataset directory")
parser.add_argument("-f", nargs="*", type=str, metavar="list_name")
parser.add_argument("-v", nargs="*", type=int, metavar="fold")
args = parser.parse_args()

rootpath, filelist, fold = check_option(args)
for idx, label in enumerate(listdir(rootpath)):
    dirpath = join(rootpath, label)
    pathlist = [p for p in listdir(dirpath) if isfile(join(dirpath, p))]
    random.shuffle(pathlist)

    num = [v*len(pathlist)//sum(fold) for v in fold]
    num = [n+1 if i < len(pathlist)-sum(num) else n for i, n in enumerate(num)]
    acc = list(accumulate(num))
    for i, f in enumerate(filelist):
        for image in pathlist[acc[i]-num[i]:acc[i]]:
            print(join(dirpath, image) + " " + str(idx+1), file=f)
