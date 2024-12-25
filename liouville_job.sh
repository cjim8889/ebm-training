#!/bin/bash

#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o liouville_output.log
#PBS -M wc5118@ic.ac.uk                                               
#PBS -m bae

cd $PBS_O_WORKDIR


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate py12

python main.py \
  --num-epochs=8000 \
  --batch-size=256 \
  --num-samples=5120 \
  --num-steps=200 \
  --learning-rate=1e-3 \
  --hidden-dim=256 \
  --depth=4 \
  --num-timesteps=128 \
  --mcmc-steps=5 \
  --mcmc-integration-steps=10 \
  --eta=0.05 \
  --schedule="power" \
  --integrator="euler" \
  --optimizer="adamw" \
  --target="lj13bt" \
  --seed=97979 \
  --continuous-schedule \
  --soft-clip \
  --target-end-time 1. \
  --offline

# CUDA_VISIBLE_DEVICES=3 uv run train_modified_liouville.py --hidden_dim 256 --N 1024 --T 128 --schedule inverse_power --input_dim 10 --depth 3 --mcmc_type langevin --num_mcmc_steps 5 --num_mcmc_integration_steps 5 --eta 1.0 --num_epochs 16000 --num_steps 50
