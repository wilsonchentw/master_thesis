#!/bin/bash

DATASET=50data
LIBSVM_PATH=../libsvm
LIBLINEAR_PATH=../liblinear

TRAIN=${DATASET}_train
VAL=${DATASET}_val
TEST=${DATASET}_test

python3 split_dataset.py ../$DATASET\
    -f ${TRAIN}.list ${VAL}.list ${TEST}.list\
    -v 1 4 1

g++ extract_features.cpp $(pkg-config --cflags --libs opencv) -o extract_features
./extract_features ${TRAIN}.list ${TRAIN}_raw.dat
./extract_features ${TEST}.list ${TEST}_raw.dat

${LIBLINEAR_PATH}/train -c 1 -q ${TRAIN}_raw.dat ${DATASET}_raw.model
${LIBLINEAR_PATH}/predict ${TRAIN}_raw.dat ${DATASET}_raw.model ${TRAIN}.predict
${LIBLINEAR_PATH}/predict ${TEST}_raw.dat  ${DATASET}_raw.model ${TEST}.predict
${LIBLINEAR_PATH}/train -c 0.1 -q ${TRAIN}_raw.dat ${DATASET}_raw.model
${LIBLINEAR_PATH}/predict ${TRAIN}_raw.dat ${DATASET}_raw.model ${TRAIN}.predict
${LIBLINEAR_PATH}/predict ${TEST}_raw.dat  ${DATASET}_raw.model ${TEST}.predict
${LIBLINEAR_PATH}/train -c 0.01 -q ${TRAIN}_raw.dat ${DATASET}_raw.model
${LIBLINEAR_PATH}/predict ${TRAIN}_raw.dat ${DATASET}_raw.model ${TRAIN}.predict
${LIBLINEAR_PATH}/predict ${TEST}_raw.dat  ${DATASET}_raw.model ${TEST}.predict
${LIBLINEAR_PATH}/train -c 0.001 -q ${TRAIN}_raw.dat ${DATASET}_raw.model
${LIBLINEAR_PATH}/predict ${TRAIN}_raw.dat ${DATASET}_raw.model ${TRAIN}.predict
${LIBLINEAR_PATH}/predict ${TEST}_raw.dat  ${DATASET}_raw.model ${TEST}.predict
${LIBLINEAR_PATH}/train -c 0.0001 -q ${TRAIN}_raw.dat ${DATASET}_raw.model
${LIBLINEAR_PATH}/predict ${TRAIN}_raw.dat ${DATASET}_raw.model ${TRAIN}.predict
${LIBLINEAR_PATH}/predict ${TEST}_raw.dat  ${DATASET}_raw.model ${TEST}.predict
${LIBLINEAR_PATH}/train -c 0.00001 -q ${TRAIN}_raw.dat ${DATASET}_raw.model
${LIBLINEAR_PATH}/predict ${TRAIN}_raw.dat ${DATASET}_raw.model ${TRAIN}.predict
${LIBLINEAR_PATH}/predict ${TEST}_raw.dat  ${DATASET}_raw.model ${TEST}.predict
${LIBLINEAR_PATH}/train -c 0.000001 -q ${TRAIN}_raw.dat ${DATASET}_raw.model
${LIBLINEAR_PATH}/predict ${TRAIN}_raw.dat ${DATASET}_raw.model ${TRAIN}.predict
${LIBLINEAR_PATH}/predict ${TEST}_raw.dat  ${DATASET}_raw.model ${TEST}.predict

#${LIBSVM_PATH}/svm-train -c 0.0001 -q ${TRAIN}_raw.dat ${DATASET}_raw.model
#${LIBSVM_PATH}/svm-predict  ${TRAIN}_raw.dat ${DATASET}_raw.model ${TRAIN}.predict
#${LIBSVM_PATH}/svm-predict  ${TEST}_raw.dat  ${DATASET}_raw.model ${TEST}.predict
#${LIBSVM_PATH}/tools/grid.py -gnuplot null ${TRAIN}_raw.dat 

#rm 50data* extract_features
rm extract_features
