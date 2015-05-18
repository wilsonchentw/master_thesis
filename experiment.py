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
            os.path.join(lib['spams-matlab'], "test_release"), 
            os.path.join(lib['spams-matlab'], "src_release"), 
            os.path.join(lib['spams-matlab'], "build"), 
        ])
    )


def generate_list(path, listname, percent):
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
        cwd = getcwd()
        din = normpath(args.din)
        prefix = basename(din)

        if not isdir(args.din):
            print "Input is not directory"
            exit(-1)
        elif any([isfile(f) and f.startswith(prefix) for f in listdir(cwd)]):
            print "Found possible duplicate file in `{0}`:".format(cwd)
            for f in listdir(cwd):
                if isfile(f) and f.startswith(prefix):
                    print os.path.join(cwd, f)
            exit(-1)
        else:
            print "Generate image list ... "
            generate_list(din, ["small", "medium", "large"], [5, 20, 50])
            generate_list(din, ["full"], [100])
            exit(0)
    elif args.fin is not None:
        if not isfile(args.fin):
            print "Input is not file"
            exit(-1)
        else:
            # If input is image list, setup environment variable
            root = dirname(realpath(sys.argv[0]))
            fin = realpath(args.fin)
            setup_environment(lib)

            vl_setup = os.path.join(lib['vlfeat'], "toolbox", "vl_setup.m")
            baseline = os.path.join(root, 'baseline')

            #run(fullfile(root_dir, '../vlfeat/toolbox/vl_setup'));
            cmd = "addpath('{0}'); baseline {1}; quit".format(baseline, fin)
            start_args = ["-nodesktop -nosplash -singleCompThread -r"]
            #subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)

