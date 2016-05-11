#!/bin/sh
#PBS -N regularization-optimization
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q k40_interactive
#PBS -m eba
#PBS -M lasseregin@gmail.com

cd $PBS_O_WORKDIR

# Enable python3
export PYTHONPATH=
source ~/stdpy3/bin/activate

python speaker_classifier_reg_opt.py
