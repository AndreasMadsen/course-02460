#!/bin/sh
#PBS -N regularization-optimization-2d-classifier
#PBS -l walltime=05:00:00
#PBS -l nodes=1:ppn=10
#PBS -m eba
#PBS -M lasseregin@gmail.com

cd $PBS_O_WORKDIR

# Enable python3
export PYTHONPATH=
source ~/stdpy3/bin/activate

THEANO_FLAGS='device=cpu' python 2d_classifier_reg_opt.py
