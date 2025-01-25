#!/bin/bash

#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o mw32_output.log
#PBS -M wc5118@ic.ac.uk                                               
#PBS -m bae

cd $PBS_O_WORKDIR


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate py12

python main.py \
  --num-samples 2560 \
  --depth 4 \
  --hidden-dim 128 \
  --network mlp2 \
  --num-epochs 10000 \
  --steps-per-epoch 500 \
  --mcmc-method vsmc \
  --mcmc-step-size 0.1 \
  --mcmc-steps 6 \
  --mcmc-integration-steps 10 \
  --initial-sigma 2. \
  --with-rejection \
  --target mw32 \
  --seed 1234 \
  --use-decoupled-loss \
  --batch-size 128 \
  --learning-rate 1e-03 \
  --gradient-norm 1. \
  --optimizer adamw \
  --weight-decay 1e-04 \
  --time-batch-size 64 \
  --n-samples-eval 1024 \
  --use-cv \
  --eval-frequency 60 \
  --use-shortcut \
  --shortcut-size 32 64 128 \
  --use-hutchinson \
  --continuous-time \
  --n-probes 1

