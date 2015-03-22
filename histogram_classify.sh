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

${LIBLINEAR_PATH}/train   -c 0.1 ${TRAIN}_hist.dat ${DATASET}_hist.model
${LIBLINEAR_PATH}/predict ${TRAIN}_hist.dat ${DATASET}_hist.model ${TRAIN}.predict
${LIBLINEAR_PATH}/predict ${TEST}_hist.dat  ${DATASET}_hist.model ${TEST}.predict

rm 50data* extract_hist
