#!/bin/bash

DATASET=50data
LIBSVM_PATH=../libsvm
LIBLINEAR_PATH=../liblinear

TRAIN=${DATASET}_train
VAL=${DATASET}_val
TEST=${DATASET}_test

python3 split_dataset.py ../$DATASET\
    -f ${TRAIN}.list ${VAL}.list ${TEST}.list\
    -v 1 0 0
#    -v 1 98 1

python extract_features.py ${TRAIN}.list ${TRAIN}.dat
#python extract_features.py ${TEST}.list ${TEST}.dat

${LIBLINEAR_PATH}/train -c 1 -v 5 -q ${TRAIN}.dat ${DATASET}.linear.model
${LIBLINEAR_PATH}/train -c 0.1 -v 5 -q ${TRAIN}.dat ${DATASET}.linear.model
${LIBLINEAR_PATH}/train -c 10 -v 5 -q ${TRAIN}.dat ${DATASET}.linear.model

${LIBSVM_PATH}/svm-train -v 5 -q ${TRAIN}.dat ${DATASET}.model
#${LIBSVM_PATH}/tools/grid.py -gnuplot null ${TRAIN}.dat 

#${LIBLINEAR_PATH}/predict ${TRAIN}.dat ${DATASET}.linear.model ${TRAIN}.predict
#${LIBLINEAR_PATH}/predict ${TEST}.dat  ${DATASET}.linear.model ${TEST}.predict
#${LIBSVM_PATH}/svm-predict ${TRAIN}.dat ${DATASET}.model ${TRAIN}.predict
#${LIBSVM_PATH}/svm-predict ${TEST}.dat  ${DATASET}.model ${TEST}.predict

