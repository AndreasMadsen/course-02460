#!/bin/sh
#PBS -N regularization-optimization-geometry-offset
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=8
#PBS -m eba
#PBS -M lasseregin@gmail.com

cd $PBS_O_WORKDIR

# Enable python3
export PYTHONPATH=
source ~/stdpy3/bin/activate

python geometry_classifier.py
