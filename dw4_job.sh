#!/bin/bash

#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o dw4_output.log
#PBS -M wc5118@ic.ac.uk                                               
#PBS -m bae

cd $PBS_O_WORKDIR


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate py12

python main.py \
  --num-samples 5120 \
  --depth 4 \
  --hidden-dim 256 \
  --network pdn2 \
  --num-epochs 20000 \
  --steps-per-epoch 300 \
  --mcmc-method vsmc \
  --mcmc-step-size 0.1 \
  --mcmc-steps 10 \
  --mcmc-integration-steps 10 \
  --initial-sigma 2. \
  --with-rejection \
  --target dw4o \
  --seed 12345 \
  --use-decoupled-loss \
  --batch-size 128 \
  --learning-rate 1e-03 \
  --gradient-clip 1. \
  --optimizer adamw \
  --weight-decay 1e-03 \
  --time-batch-size 64 \
  --n-samples-eval 1024 \
  --use-cv \
  --data-path-test data/test_split_DW4.npy \
  --eval-frequency 60 \
  --use-shortcut \
  --shortcut-weight 0.01 \
  --use-hutchinson \
  --every-k-schedule 1 \
  --mixed-precision \
  --annealing-path inverse_power \
  --use-schedule \
  --use-combined-loss

