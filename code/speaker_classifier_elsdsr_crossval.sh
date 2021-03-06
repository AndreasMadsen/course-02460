#!/bin/sh
#PBS -N speaker-classifier-elsdsr-crossval
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
python speaker_classifier_elsdsr_crossval.py
