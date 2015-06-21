#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from os import environ, pathsep, getenv
from os.path import realpath, basename, dirname, expanduser
from pprint import pprint
import sys
import subprocess


lib = {
    'gcc': "/usr/lib/gcc/x86_64-linux-gnu/4.8", 
    'vlfeat': "~/Software/vlfeat", 
    'spams-matlab': "~/Software/spams-matlab", 
    'spams-python': (
        "~/Software/spams-python/install/lib/python2.7/site-packages"
    ), 
    'liblinear': "~/Software/liblinear", 
    'libsvm': "~/Software/libsvm", 
}


def setup_3rdparty(lib):

    # Normalize library path
    for name, path in lib.items():
        lib[name] = realpath(expanduser(path))

    # Setup environment variable
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
            os.path.join(lib['spams-matlab'], "src_release"), 
            os.path.join(lib['spams-matlab'], "test_release"), 
        ])
    )

    return lib


def setup_path(root, paths):
    paths = [os.path.join(root, path) for path in paths]
    paths.append(root)
    if getenv('MATLABPATH') is not None:
        paths.insert(0, getenv('MATLABPATH'))

    environ['MATLABPATH'] = pathsep.join(paths)


def generate_list(path, listname, percent):

    root = dirname(realpath(sys.argv[0]))
    path = realpath(path)
    prefix = basename(path)

    listname = ["{0}_{1}.list".format(prefix, name) for name in listname]
    if sum(percent) < 100:
        listname.append(os.devnull)
        percent.append(100 - sum(percent))

    split_dataset = os.path.join(root, "util", "split_dataset.py")
    cmd = (
        ["python", split_dataset, path] + 
        ["-f"] + listname + 
        ["-v"] + map(str, percent)
    )
    subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser()
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument('-f', metavar="image_list", dest="fin")
    dataset.add_argument('-d', metavar="images_dir", dest="din")
    args = parser.parse_args()


    if args.din is not None:
        din = expanduser(args.din)

        generate_list(din, ["train", "val", "test"], [64, 16, 20])
        generate_list(din, ["full"], [100])
        exit(0)
    else:
        fin = realpath(expanduser(args.fin))
        root = dirname(realpath(sys.argv[0]))

        # Setup third-party library path
        lib = setup_3rdparty(lib)
        setup_path(root, ["util", "thesis", "baseline"])

        vl_setup = os.path.join(lib['vlfeat'], "toolbox", "vl_setup")
        vl_setup = "run('{0}')".format(vl_setup)
        run_thesis = "run_thesis('{0}')".format(fin)

        start_args = "-nodesktop -nosplash -singleCompThread"
        matlab_cmd = "; ".join([vl_setup, run_thesis])
        cmd = ["matlab", start_args, "-r", matlab_cmd]
        subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        
"""
def run_baseline(fin):
    # Run baseline
    vl_script = os.path.join(lib['vlfeat'], "toolbox", "vl_setup")
    setup_vl = "run('{0}')".format(vl_script)
    setup_baseline = "addpath('{0}')".format(os.path.join(root, 'baseline'))
    run_baseline = "run_baseline '{0}'".format(fin)

    start_args = "-nodesktop -nosplash -singleCompThread -r"
    matlab_cmd = "; ".join([setup_vl, setup_baseline, run_baseline, 'quit'])
    cmd = ["matlab", start_args, '"{0}"'.format(matlab_cmd)]
    subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)

# Run thesis in python
#cmd = ["python", os.path.join(root, "thesis", "run_thesis.py"), fin]
#subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)
"""
