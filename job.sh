#!/bin/bash

#PBS -N ebm_cifar10
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o ebm_cifar10_output.log
#PBS -M wc5118@ic.ac.uk                                               
#PBS -m bae

cd $PBS_O_WORKDIR


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate torch_cuda_env

CUDA_VISIBLE_DEVICES=0 uv run main.py  --num-samples 5120 \
  --depth 4 \
  --hidden-dim 128 \
  --network mlp \
  --num-epochs 10000 \
  --steps-per-epoch 200 \
  --mcmc-method vsmc \
  --mcmc-step-size 0.2 \
  --mcmc-steps 5 \
  --mcmc-integration-steps 5 \
  --initial-sigma 20. \
  --with-rejection \
  --target gmm \
  --seed 888 \
  --use-decoupled-loss \
  --batch-size 128 \
  --learning-rate 6e-04 \
  --gradient-norm 1. \
  --optimizer adamw \
  --weight-decay 1e-04 \
  --time-batch-size 64
# Deactivate the virtual environment
deactivate