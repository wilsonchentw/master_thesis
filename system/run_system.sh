#!/bin/sh

#export LIB_GCC=/usr/lib/gcc/x86_64-linux-gnu/4.8
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_GCC:$LIB_VLFEAT:/
#export LD_PRELOAD=$LIB_GCC/libgfortran.so:$LIB_GCC/libgcc_s.so
#export LD_PRELOAD=$LD_PRELOAD:$LIB_GCC/libstdc++.so:/$LIB_GCC/libgomp.so

if [ "$#" -eq 1 ]; then
    matlab_option='-nosplash -nojvm -singleCompThread -r'

    #matlab_cmd="addpath('matlab'); run_train('$1'); quit;"
    matlab_cmd="addpath('matlab'); run_demo('$1'); quit;"

    matlab $matlab_option "$matlab_cmd" | tail -n +11
fi
