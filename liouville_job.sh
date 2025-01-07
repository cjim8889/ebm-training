#!/bin/bash

#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o liouville_output.log
#PBS -M wc5118@ic.ac.uk                                               
#PBS -m bae

cd $PBS_O_WORKDIR


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate py12

python main.py \
  --num-samples 5120 \
  --depth 5 \
  --hidden-dim 512 \
  --network mlp \
  --num-epochs 10000 \
  --steps-per-epoch 20 \
  --mcmc-method vsmc \
  --mcmc-step-size 0.02 \
  --mcmc-steps 5 \
  --mcmc-integration-steps 10 \
  --initial-sigma 1. \
  --with-rejection \
  --target sclj13 \
  --seed 888 \
  --alpha 0.4 \
  --include-harmonic \
  --log-prob-clip 100 \
  --use-decoupled-loss


