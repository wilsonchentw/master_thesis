#!/bin/bash

DATASET=50data
LIBSVM_PATH=../libsvm
LIBLINEAR_PATH=../liblinear

TRAIN=${DATASET}_train
VAL=${DATASET}_val
TEST=${DATASET}_test

rm 50data_*

python3 split_dataset.py ../$DATASET\
    -f ${TRAIN}.list ${VAL}.list ${TEST}.list\
    -v 4 1 1

g++ extract_patch.cpp $(pkg-config --cflags --libs opencv) -o extract_patch
./extract_patch ${TRAIN}.list ${TRAIN}_patch.dat
./extract_patch ${VAL}.list   ${VAL}_patch.dat
./extract_patch ${TEST}.list  ${TEST}_patch.dat

## Clean
rm extract_patch
ls -lah --color
