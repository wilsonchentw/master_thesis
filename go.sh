#!/bin/bash

g++ preprocess.cpp $(pkg-config --cflags --libs opencv) -o preprocess
python3 split_dataset.py ../50data train_list test_list 5
./preprocess train_list train_libsvm
rm preprocess
