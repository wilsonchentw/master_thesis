@echo off
matlab -nosplash -nojvm -singleCompThread -wait -r "addpath('matlab'); run_demo('%1', '%2'); quit;"
type %2