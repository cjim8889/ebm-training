#!/bin/bash

#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o lj13_output.log
#PBS -M wc5118@ic.ac.uk                                               
#PBS -m bae

cd $PBS_O_WORKDIR


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate py12

python main.py \
  --num-samples 2560 \
  --depth 6 \
  --hidden-dim 128 \
  --network transformer5 \
  --num-epochs 20000 \
  --steps-per-epoch 1000 \
  --mcmc-method vsmc \
  --mcmc-step-size 0.02 \
  --mcmc-steps 15 \
  --mcmc-integration-steps 10 \
  --initial-sigma 2. \
  --with-rejection \
  --target smlj13q \
  --seed 12345 \
  --use-decoupled-loss \
  --batch-size 128 \
  --learning-rate 1e-04 \
  --gradient-norm 1. \
  --optimizer adamw \
  --weight-decay 1e-06 \
  --time-batch-size 4 \
  --n-samples-eval 1024 \
  --eval-frequency 10 \
  --shortcut-weight 0.1 \
  --include-harmonic \
  --estimator none \
  --r-min 0.8 \
  --num-heads 4 \
  --lambda-epochs 100 \
  --ess-threshold 0.5 \
  --perturb --perturbation-scale 1.0 \
  --log-z-estimation-frequency 1 \
  --augment --translation-scale 10. --continuous-time \
  --use-shortcut 


