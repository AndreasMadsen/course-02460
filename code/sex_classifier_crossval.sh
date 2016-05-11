#!/bin/sh
#PBS -N sex-classifier-crossval
#PBS -l walltime=03:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -m eba
#PBS -M amwebdk@gmail.com
#PBS -q k40_interactive

cd $PBS_O_WORKDIR

# Enable python3
export PYTHONPATH=
source ~/stdpy3/bin/activate

echo $HOME
python sex_classifier_crossval.py
