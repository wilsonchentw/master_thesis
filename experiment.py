#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from os import environ, pathsep
from os import getenv, listdir, getcwd
from os.path import realpath, isfile, dirname
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
    for name, path in lib.items():
        lib[name] = realpath(path)

    # Setup environment variable
    environ['LD_LIBRARY'] = pathsep.join(
        filter(None, [
            getenv('LD_LIBRARY'), 
            lib['gcc'], 
            os.path.join(lib['vlfeat'], "bin", "glnxa64", "libvl.so"),
        ])
    )
    environ['LD_PRELOAD'] = pathsep.join(
        filter(None, [
            getenv('LD_PRELOAD'), 
            os.path.join(lib['gcc'], "libgfortran.so"), 
            os.path.join(lib['gcc'], "libgcc_s.so"), 
            os.path.join(lib['gcc'], 'libstdc++.so'), 
            os.path.join(lib['gcc'], 'libgomp.so'), 
        ])
    )
    environ['PYTHONPATH'] = pathsep.join(
        filter(None, [
            getenv('PYTHON'), 
            os.path.join(lib['libsvm'], "python"), 
            os.path.join(lib['liblinear'], "python"), 
            os.path.join(lib['spams_py']), 
        ])
    )


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
            print "Generate image list ..."
            generate_list(args.din, ["small", "medium", "large"], [5, 20, 50])
            generate_list(args.din, ["full"], [100])
            exit(0)


    # If input is image list, setup environment variable
    setup_environment(lib)
    root = dirname(realpath(sys.argv[0]))
