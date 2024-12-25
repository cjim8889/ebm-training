#!/bin/bash

#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o shortcut_liouville_output.log
#PBS -M wc5118@ic.ac.uk                                               
#PBS -m bae

cd $PBS_O_WORKDIR


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate py12

python train_shortcut_liouville.py \
  --num-epochs=8000 \
  --batch-size=256 \
  --shortcut-size=64 \
  --num-samples=5120 \
  --num-steps=200 \
  --learning-rate=1e-3 \
  --hidden-dim=256 \
  --depth=4 \
  --num-timesteps=128 \
  --mcmc-steps=5 \
  --mcmc-integration-steps=3 \
  --eta=1.0 \
  --schedule="inverse_power" \
  --integrator="euler" \
  --optimizer="adamw" \
  --seed=97979 \
  --d-distribution="log" \
  --eval-steps 4 8 16 32 64 \
  --continuous-schedule

# Deactivate the virtual environment
deactivate
