#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import random
import argparse
from os import getcwd, listdir
from os.path import isfile, join, abspath

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="dataset directory", type=str)
parser.add_argument("train_list", help="training set filelist", type=str)
parser.add_argument("test_list", help="testing set filelist", type=str)
parser.add_argument("fold", help="n-fold validation", type=int)
if len(sys.argv) <= 1:
    args = parser.parse_args(['-h'])
else:
    args = parser.parse_args()

v = args.fold
rootpath = abspath(args.dataset)
trainfile = open(args.train_list, mode='w')
testfile = open(args.test_list, mode='w')
for idx, label in enumerate(listdir(rootpath)):
    dirpath = join(rootpath, label)
    filelist = [ f for f in listdir(dirpath) if isfile(join(dirpath, f)) ]
    random.shuffle(filelist)
    test_sample = filelist[0:len(filelist):v]
    train_sample = [f for f in filelist if f not in test_sample]
    for f in test_sample: print(join(dirpath, f) + " " + str(idx), file=testfile)
    for f in train_sample: print(join(dirpath, f) + " " + str(idx), file=trainfile)
