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
  --num-steps=100 \
  --learning-rate=1e-3 \
  --hidden-dim=256 \
  --depth=4 \
  --num-timesteps=128 \
  --mcmc-steps=5 \
  --mcmc-integration-steps=10 \
  --eta=0.05 \
  --initial-sigma=2.0 \
  --schedule="linear" \
  --integrator="euler" \
  --optimizer="adamw" \
  --target="lj13bt" \
  --seed=97979 \
  --continuous-schedule \
  --soft-clip \
  --target-end-time 1. \
  --pt-clip 500 

