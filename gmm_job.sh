#!/bin/bash

#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o output_gmm.log
#PBS -M wc5118@ic.ac.uk                                               
#PBS -m bae

cd $PBS_O_WORKDIR


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate py12

python main.py \
  --num-samples 5120 \
  --depth 4 \
  --hidden-dim 256 \
  --network mlp \
  --num-epochs 20000 \
  --steps-per-epoch 500 \
  --mcmc-method vsmc \
  --mcmc-step-size 0.1 \
  --mcmc-steps 5 \
  --mcmc-integration-steps 5 \
  --initial-sigma 25. \
  --with-rejection \
  --target gmm \
  --seed 123 \
  --use-decoupled-loss \
  --batch-size 128 \
  --learning-rate 6e-04 \
  --gradient-norm 1. \
  --optimizer adamw \
  --weight-decay 1e-04 \
  --time-batch-size 32 \
  --n-samples-eval 1024 \
  --use-cv \
  --eval-frequency 100 \
  --use-shortcut

