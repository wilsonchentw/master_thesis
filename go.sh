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
 
${LIBSVM_PATH}/svm-scale -l 0 -u 1 -s ${TRAIN}.range ${TRAIN}.dat > ${TRAIN}.scale.dat
${LIBSVM_PATH}/svm-scale -l 0 -u 1 -r ${TRAIN}.range ${VAL}.dat   > ${VAL}.scale.dat &
${LIBSVM_PATH}/svm-scale -l 0 -u 1 -r ${TRAIN}.range ${TEST}.dat  > ${TEST}.scale.dat &

python ${LIBSVM_PATH}/tools/grid.py ${TRAIN}.scale.dat

rm preprocess
ls --color -lah
