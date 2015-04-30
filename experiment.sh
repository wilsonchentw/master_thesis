#!/bin/bash

LIBSVM=../libsvm
LIBLINEAR=../liblinear
SPAMS_PYTHON=../spams/spams-python/install/lib/python2.7/site-packages/

# Export environment variable for SPAMS, libsvm and liblinear
export LIB_GCC=/usr/lib/gcc/x86_64-linux-gnu/4.8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_GCC:/
export LD_PRELOAD=$LIB_GCC/libgfortran.so:$LIB_GCC/libgcc_s.so
export LD_PRELOAD=$LD_PRELOAD:$LIB_GCC/libstdc++.so:/$LIB_GCC/libgomp.so
export PYTHONPATH=$PYTHONPATH:$(realpath $LIBSVM/python)
export PYTHONPATH=$PYTHONPATH:$(realpath $LIBLINEAR/python)
export PYTHONPATH=$PYTHONPATH:$(realpath $SPAMS_PYTHON)

# Parse script parameter
if [[ $# -ne 1 ]] || [[ ! -f $1 && ! -d $1 ]]; then
  echo "Generate image list of dataset_dir or do experement on image_list"
  echo "Usage: $0 dataset_dir"
  echo "       $0 image_list"
  exit -1
fi

# If input is directory, genereate image list
if [[ -d $1 ]]; then
  dataset_path=$1
  dataset=$(basename $dataset_path)
  duplicate=$(ls -1 ${dataset}*.list 2> /dev/null | wc -l)
  if [[ $duplicate != 0 ]]; then
    echo "Duplicate file exist, please check your file."
    exit -1
  else
    echo -n "Generate image list with different size ... "
    python3 python/split_dataset.py $dataset_path\
        -f ${dataset}_full.list -v 100
    python3 python/split_dataset.py $dataset_path\
        -f ${dataset}_small.list ${dataset}_medium.list ${dataset}_large.list\
        -v 5 20 75
    echo "done !"
    exit 0
  fi
fi

image_list=$1
script_dir=$(cd "$(dirname "$0")" && pwd)

# Start baseline method
#setup_cmd="addpath(fullfile('$script_dir', 'matlab'))"
#setup_cmd=$setup_cmd"; setup_3rdparty('$script_dir')"
#matlab_cmd=$setup_cmd"; baseline $image_list"
#matlab -nodesktop -nosplash -singleCompThread -r "$matlab_cmd; quit"

# Start my method
python python/thesis.py $image_list
