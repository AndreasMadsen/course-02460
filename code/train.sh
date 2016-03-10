#!/bin/sh
#PBS -N single-model
# -- email me at the beginning (b) and end (e) of the execution --
#PBS -m e
#PBS -M s123815@student.dtu.dk
#PBS -l walltime=40:00:00
#PBS -q k40_interactive
#PBS -l nodes=1:ppn=4:gpus=1

module load python3
module load gcc
module load qt
module load cuda
export PYTHONPATH=
source ~/stdpy3/bin/activate

if [ -z "$PBS_JOBID" ]; then
	echo "PBS_JOBID is unset";
else
	echo "PBS_JOBID is set to '$PBS_JOBID'";
	cd $PBS_O_WORKDIR
fi

# Define output file
outputfile="output.txt"

# Create empty output file
echo "" > $outputfile

# Train model
echo "Training model.." >> $outputfile
#THEANO_FLAGS="floatX=float64,device=$device"
#THEANO_FLAGS="floatX=float32,device=$device"
THEANO_FLAGS='floatX=float64,device=gpu0,lib.cnmem=1' python3 main.py >> $outputfile
#python3 train.py -model $model -learning_model $learning_model -epochs $epochs -learning_rate $learning_rate -data_set $data_set -load_from_file $load_from_file -batch_size $batch_size -min_word_count $min_count -use_word2vec $use_word2vec -word2vec_vocabsize $word2vec_vocabsize >> $outputfile
echo "" >> $outputfile
