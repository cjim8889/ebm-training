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
  --network transformer5 \
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
  --learning-rate 1e-04 \
  --gradient-norm 1. \
  --optimizer adamw \
  --weight-decay 1e-04 \
  --time-batch-size 16 \
  --eval-frequency 25 \
  --n-samples-eval 1024 \
  --eval-frequency 60 \
  --continuous-time \
  --estimator hutchinson \
  --n-probes 2 \
  --lambda-epochs 1000 \
  --ess-threshold 0.5 \
  --perturb --perturbation-scale 1.0 \
  --log-z-estimation-frequency 1
  

