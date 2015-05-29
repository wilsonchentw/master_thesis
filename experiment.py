#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from os import environ, pathsep
from os import getenv, listdir, getcwd
from os.path import normpath, realpath, basename, dirname, isfile, isdir
import subprocess
import sys

lib = {
    'gcc': "/usr/lib/gcc/x86_64-linux-gnu/4.9", 
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
        lib[name] = realpath(normpath(os.path.expanduser(path)))

    # Setup environment variable
    #environ['LD_LIBRARY_PATH'] = pathsep.join(
    #    filter(None, [
    #        getenv('LD_LIBRARY_PATH'), 
    #        lib['gcc'], 
    #    ])
    #)
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


def run_baseline(fin):
    # Run baseline
    vl_script = os.path.join(lib['vlfeat'], "toolbox", "vl_setup")
    setup_vl = "run('{0}')".format(vl_script)
    setup_baseline = "addpath('{0}')".format(os.path.join(root, 'baseline'))
    run_baseline = "baseline '{0}'".format(fin)

    start_args = "-nodesktop -nosplash -singleCompThread -r"
    matlab_cmd = "; ".join([setup_vl, setup_baseline, run_baseline, 'quit'])
    cmd = ["matlab", start_args, '"{0}"'.format(matlab_cmd)]
    subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser()
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument('-f', metavar="image_list", dest="fin")
    dataset.add_argument('-d', metavar="images_dir", dest="din")


    # Parse argument
    args = parser.parse_args()
    fin = realpath(normpath(args.fin))

    # Setup root & third-party library
    lib = setup_3rdparty(lib)
    root = dirname(realpath(normpath(sys.argv[0])))

    # Run baseline method
    run_baseline(fin)

    cmd = ["python", os.path.join(root, "thesis", "run_thesis.py"), fin]
    subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)
