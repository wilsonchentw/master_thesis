#!/bin/bash

DATASET=50data
LIBSVM_PATH=../libsvm
LIBLINEAR_PATH=../liblinear

TRAIN=${DATASET}_train
VAL=${DATASET}_val
TEST=${DATASET}_test

# Export environment variable for SPAMS(SPArse Modeling Software) path
export LIB_GCC=/usr/lib/gcc/x86_64-linux-gnu/4.8
export MKL_NUM_THREADS=1
export MKL_SERIAL=YES
export MKL_DYNAMIC=NO
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/gcc/x86_64-linux-gnu/4.8/:/
export LD_PRELOAD=$LIB_GCC/libgfortran.so:$LIB_GCC/libgcc_s.so
export LD_PRELOAD=$LD_PRELOAD:$LIB_GCC/libstdc++.so:/$LIB_GCC/libgomp.so

MATLAB_CMD="addpath(fullfile('./matlab')); setup_3rdparty"
MATLAB_CMD=$MATLAB_CMD'; feature_classify '${TRAIN}.list'; '
matlab -nodesktop -nosplash -singleCompThread -r "$MATLAB_CMD"


#python3 split_dataset.py ../$DATASET\
#    -f ${TRAIN}.list ${VAL}.list ${TEST}.list\
#    -v 1 0 0
#python extract_features.py ${TRAIN}.list ${TRAIN}.dat
#python extract_features.py ${TEST}.list ${TEST}.dat
#
#${LIBLINEAR_PATH}/train -c 10 -v 5 -q ${TRAIN}.dat ${DATASET}.linear.model
#${LIBLINEAR_PATH}/train -c 1 -v 5 -q ${TRAIN}.dat ${DATASET}.linear.model
#${LIBLINEAR_PATH}/train -c 0.1 -v 5 -q ${TRAIN}.dat ${DATASET}.linear.model
#${LIBLINEAR_PATH}/train -c 0.01 -v 5 -q ${TRAIN}.dat ${DATASET}.linear.model
#${LIBLINEAR_PATH}/train -c 0.001 -v 5 -q ${TRAIN}.dat ${DATASET}.linear.model
#
#${LIBSVM_PATH}/svm-train -v 5 -q ${TRAIN}.dat ${DATASET}.model
#${LIBSVM_PATH}/tools/grid.py -gnuplot null ${TRAIN}.dat 
#
#${LIBLINEAR_PATH}/predict ${TRAIN}.dat ${DATASET}.linear.model ${TRAIN}.predict
#${LIBLINEAR_PATH}/predict ${TEST}.dat  ${DATASET}.linear.model ${TEST}.predict
#${LIBSVM_PATH}/svm-predict ${TRAIN}.dat ${DATASET}.model ${TRAIN}.predict
#${LIBSVM_PATH}/svm-predict ${TEST}.dat  ${DATASET}.model ${TEST}.predict
