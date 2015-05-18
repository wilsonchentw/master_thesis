#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from os import environ as env
from os import pathsep as sep
from os import getenv, listdir, getcwd
from os.path import abspath, isfile
import subprocess
import sys


# Setup variable
lib = {
    'gcc': "/usr/lib/gcc/x86_64-linux-gnu/4.8", 
    'vlfeat': "../vlfeat", 
    'libsvm': "../libsvm", 
    'liblinear': "../liblinear", 
    'spams_py': "../spams/spams-python/install/lib/python2.7/site-packages", 
}


def setup_environment(lib):
    # Setup environment variable
    env['LD_LIBRARY_PATH'] = sep.join([
        getenv('LD_LIBRARY_PATH', ""), 
        lib['gcc'], 
        lib['vlfeat'] + "/bin/glnxa64/libvl.so", 
    ])
    env['LD_PRELOAD'] = sep.join([
        getenv('LD_PRELOAD', ""), 
        lib['gcc'] + "/libgfortran.so", 
        lib['gcc'] + "/libgcc_s.so", 
        lib['gcc'] + "/libstdc++.so", 
        lib['gcc'] + "/libgomp.so", 
    ])
    env['PYTHONPATH'] = sep.join([
        getenv('PYTHONPATH', ""), 
        lib['libsvm'] + "/python", 
        lib['liblinear'] + "/python", 
        lib['spams_py'], 
    ])


def generate_list(path, listname, percent):
    prefix = path.rpartition(os.path.sep)[2]
    listname = ["{0}_{1}.list".format(prefix, name) for name in listname]
    if sum(percent) < 100:
        listname.append(os.devnull)
        percent.append(100 - sum(percent))
 
    split_dataset = os.path.join(os.getcwd(), "split_dataset.py")
    cmd = [split_dataset, path, "-f"] + listname + ["-v"] + map(str, percent)
    subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser()
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument('-f', metavar="image_list", dest="fin")
    dataset.add_argument('-d', metavar="images_dir", dest="din")

    args = parser.parse_args()
    if args.din is not None:
        dataset = args.din.rpartition(os.path.sep)[2]
        # Check if there is possible duplicate file by prefix of filename.
        if any([isfile(f) and f.startswith(dataset) for f in listdir(".")]):
            print "Find possible duplicate file, check your working directory"
            exit(-1)
        else:
            generate_list(args.din, ["small", "medium", "large"], [5, 20, 50])
            generate_list(args.din, ["full"], [100])
    else:
        print "input is file"
        

    # Convert to absolute path, then setup environment variable
    for name, path in lib.items():
        lib[name] = abspath(path)
    setup_environment(lib)
