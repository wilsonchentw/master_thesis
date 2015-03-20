#!/bin/bash

DATASET=50data
LIBSVM_PATH=../libsvm
LIBLINEAR_PATH=../liblinear

TRAIN=${DATASET}_train
VAL=${DATASET}_val
TEST=${DATASET}_test

# Clean
rm 50data_*
rm extract_patch

python3 split_dataset.py ../$DATASET\
    -f ${TRAIN}.list ${VAL}.list ${TEST}.list\
    -v 1 98 1

g++ extract_patch.cpp $(pkg-config --cflags --libs opencv) -o extract_patch
./extract_patch ${TRAIN}.list ${TRAIN}_patch.dat
./extract_patch ${TEST}.list  ${TEST}_patch.dat

${LIBLINEAR_PATH}/train -s 2 -c 100 ${TRAIN}_patch.dat ${DATASET}_patch.model
${LIBLINEAR_PATH}/predict ${TRAIN}_patch.dat ${DATASET}_patch.model ${TRAIN}_patch.predict
${LIBLINEAR_PATH}/predict ${TEST}_patch.dat  ${DATASET}_patch.model ${TEST}_patch.predict

#ls -lah --color
