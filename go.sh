#!/bin/bash

DATASET=50data
LIBSVM_PATH=../libsvm
LIBLINEAR_PATH=../liblinear

TRAIN=${DATASET}_train
VAL=${DATASET}_val
TEST=${DATASET}_test

python3 split_dataset.py ../$DATASET\
    -f ${TRAIN}.list ${VAL}.list ${TEST}.list\
    -v 4 1 1

g++ preprocess.cpp $(pkg-config --cflags --libs opencv) -o preprocess
./preprocess ${TRAIN}.list ${TRAIN}.dat &
./preprocess ${VAL}.list ${VAL}.dat &
./preprocess ${TEST}.list ${TEST}.dat &
wait

cat ${TRAIN}.dat ${VAL}.dat > ${TRAIN}.dat
rm ${VAL}.dat
python ${LIBSVM_PATH}/tools/grid.py ${TRAIN}.dat

## Clean
rm preprocess
ls --color -lah
