#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from os import environ, pathsep
from os import getenv, listdir, getcwd
from os.path import normpath, realpath, basename, dirname, isfile, isdir
import subprocess
import sys


# Setup variable
lib = {
    'gcc': "/usr/lib/gcc/x86_64-linux-gnu/4.8", 
    'vlfeat': "../vlfeat", 
    'libsvm': "../libsvm", 
    'liblinear': "../liblinear", 
    'spams-python': (
        "../spams/spams-python/install/lib/python2.7/site-packages"
    ), 
    'spams-matlab': "../spams/spams-matlab"
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
            os.path.join(lib['gcc'], "libstdc++.so"), 
            os.path.join(lib['gcc'], "libgomp.so"), 
        ])
    )
    environ['PYTHONPATH'] = pathsep.join(
        filter(None, [
            getenv('PYTHON'), 
            os.path.join(lib['libsvm'], "python"), 
            os.path.join(lib['liblinear'], "python"), 
            os.path.join(lib['spams-python']), 
        ])
    )
    environ['MATLABPATH'] = pathsep.join(
        filter(None, [
            getenv('MATLABPATH'), 
            os.path.join(lib['liblinear'], "matlab"), 
            os.path.join(lib['libsvm'], "matlab"), 
            os.path.join(lib['spams-matlab'], "build"), 
            os.path.join(lib['spams-matlab'], "test_release"), 
            os.path.join(lib['spams-matlab'], "src_release"), 
        ])
    )


def is_valid_dir(din):  
    cwd = getcwd()
    din = realpath(normpath(din))
    prefix = basename(din)

    if not isdir(din):
        print "Input is not directory"
        return False
    elif any(isfile(f) and f.startswith(prefix) for f in listdir(cwd)):
        print "Found possible duplicate file: "
        for f in listdir(cwd):
            if isfile(f) and f.startswith(prefix):
                print "{0}".format(f)
        return False
    else:
        return True


def generate_list(path, listname, percent):
    prefix = basename(realpath(normpath(path)))
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

    # Parse argument
    args = parser.parse_args()
    if (args.din is not None) and is_valid_dir(args.din):
        print "Generate image list ... "
        args.din = realpath(normpath(args.din))
        generate_list(args.din, ["small", "medium"], [5, 20])
        generate_list(args.din, ["full"], [100])
        exit(0)
    elif (args.fin is not None) and isfile(args.fin):
        root = dirname(realpath(sys.argv[0]))
        setup_environment(lib)
    else:
        print "... Fail on running script"
        exit(-1)


    # Setup command to feed MATLAB
    fin = realpath(normpath(args.fin))
    vl_setup = os.path.join(lib['vlfeat'], "toolbox", "vl_setup")
    baseline_path = os.path.join(root, 'baseline')

    setup_vl = "run('{0}')".format(vl_setup)
    setup_baseline = "addpath('{0}')".format(baseline_path)
    run_baseline = "baseline '{0}'".format(fin)

    start_args = "-nodesktop -nosplash -singleCompThread -r"
    matlab_cmd = "\n;".join([setup_vl, setup_baseline, run_baseline])
    cmd = ["matlab", start_args, "\"{0}\"".format(matlab_cmd), ]
    subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)

