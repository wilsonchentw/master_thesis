#!/bin/sh

#export LIB_GCC=/usr/lib/gcc/x86_64-linux-gnu/4.8
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_GCC:$LIB_VLFEAT:/
#export LD_PRELOAD=$LIB_GCC/libgfortran.so:$LIB_GCC/libgcc_s.so
#export LD_PRELOAD=$LD_PRELOAD:$LIB_GCC/libstdc++.so:/$LIB_GCC/libgomp.so
export MATLABPATH=$MATLABPATH:$(pwd)/matlab

if [ "$#" -lt 1 ]; then
    image_file='test.jpg'
else [ $#  ]
    image_file=$1
fi


# Demo classification
matlab_option='-nosplash -nojvm -singleCompThread -r'
matlab_cmd="addpath('matlab'); demo('$image_file', '50data_large.mat'); quit;"
matlab $matlab_option "$matlab_cmd" | tail -n +11


## Generate model file
#matlab_option='-nosplash -nojvm -singleCompThread -r'
#matlab_cmd="addpath('matlab'); generate_model('$image_file'); quit;"
#matlab $matlab_option "$matlab_cmd" | tail -n +11
