#!/bin/bash

DATASET=50data
LIBSVM_PATH=../libsvm
LIBLINEAR_PATH=../liblinear

TINY=${DATASET}_tiny
SMALL=${DATASET}_small
MEDIUM=${DATASET}_medium
LARGE=${DATASET}_large
FULL=${DATASET}_full

# Generate different dataset size
#python3 python/split_dataset.py ../$DATASET\
#        -f ${TINY}.list ${SMALL}.list ${MEDIUM}.list ${LARGE}.list ${FULL}.list\
#        -v 3 5 20 50 22
#python3 python/split_dataset.py ../$DATASET -f ${FULL}.list -v 100

# Export environment variable for SPAMS(SPArse Modeling Software) path
export LIB_GCC=/usr/lib/gcc/x86_64-linux-gnu/4.8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/gcc/x86_64-linux-gnu/4.8/:/
export LD_PRELOAD=$LIB_GCC/libgfortran.so:$LIB_GCC/libgcc_s.so
export LD_PRELOAD=$LD_PRELOAD:$LIB_GCC/libstdc++.so:/$LIB_GCC/libgomp.so

SETUP_PATH="addpath(fullfile('./matlab')); setup_3rdparty();"
MATLAB_COMMAND=${SETUP_PATH}"baseline ${SMALL}.list"
matlab -nodesktop -nosplash -singleCompThread -r "$MATLAB_COMMAND; quit"
