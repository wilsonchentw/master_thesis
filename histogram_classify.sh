#!/bin/bash

DATASET=50data
LIBSVM_PATH=../libsvm
LIBLINEAR_PATH=../liblinear

TRAIN=${DATASET}_train
VAL=${DATASET}_val
TEST=${DATASET}_test

python3 split_dataset.py ../$DATASET\
    -f ${TRAIN}.list ${VAL}.list ${TEST}.list\
    -v 1 98 1

g++ extract_histogram.cpp $(pkg-config --cflags --libs opencv) -o extract_hist
./extract_hist ${TRAIN}.list ${TRAIN}_hist.dat
./extract_hist ${TEST}.list ${TEST}_hist.dat

rm 50data* extract_hist
