#!/bin/sh

for DATASET_IDX in 0 1 2 3
do
  for CLASSIFIER_IDX in 0 1 2
  do
    echo "qsubbing DATASET_IDX=$DATASET_IDX,CLASSIFIER_IDX=$CLASSIFIER_IDX"
    qsub -v DATASET_IDX=$DATASET_IDX,CLASSIFIER_IDX=$CLASSIFIER_IDX qsub/reg_opt_2d.sh
  done
done
