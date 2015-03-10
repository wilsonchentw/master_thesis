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

g++ im2svm.cpp $(pkg-config --cflags --libs opencv) -o im2svm
./im2svm ${TRAIN}.list ${TRAIN}.dat &
./im2svm ${VAL}.list ${VAL}.dat &
./im2svm ${TEST}.list ${TEST}.dat &
wait

cat ${TRAIN}.dat ${VAL}.dat > ${TRAIN}.dat
rm ${VAL}.dat
python ${LIBSVM_PATH}/tools/grid.py ${TRAIN}.dat

## Clean
rm im2svm
ls --color -lah
