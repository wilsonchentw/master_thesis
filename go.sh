#!/bin/bash

DATASET = 50data
FOLD = 100

g++ preprocess.cpp $(pkg-config --cflags --libs opencv) -o preprocess

#python3 split_dataset.py ../$DATASET ${DATASET}_train.list ${DATASET}_test.list
#./preprocess ${DATASET}_train.list ${DATASET}_train.libsvm
#./preprocess ${DATASET}_test.list ${DATASET}_test.libsvm

#./preprocess test_list test_libsvm
#../libsvm/svm-scale -l 0 -u 1 -s test_libsvm.param test_libsvm  > test_libsvm.scale
#../libsvm/svm-scale -r test_libsvm.param test_libsvm > test_libsvm.scale
#rm preprocess
