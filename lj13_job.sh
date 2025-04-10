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
  --embedding-dim 128 --embedder-width 128 --embedder-depth 3 \
  --network transformer8 \
  --num-epochs 20000 \
  --steps-per-epoch 1000 \
  --mcmc-method vsmc \
  --mcmc-step-size 0.01 \
  --mcmc-steps 12 \
  --mcmc-integration-steps 10 \
  --initial-sigma 2.5 \
  --target smlj13q \
  --seed 12345 \
  --training-data combined \
  --batch-size 128 \
  --learning-rate 1e-04 \
  --gradient-norm 1. \
  --optimizer adamw \
  --weight-decay 0. \
  --time-batch-size 4 \
  --n-samples-eval 5000 \
  --eval-frequency 10 \
  --shortcut-weight 0.1 \
  --include-harmonic \
  --estimator hutchinson \
  --n-probes 3 \
  --r-min 0.8 \
  --num-heads 4 \
  --lambda-epochs 500 \
  --ess-threshold 0.5 \
  --perturb --perturbation-scale .5 \
  --log-z-estimation-frequency 1 \
  --augment --translation-scale 10. --continuous-time \
  --mixed-precision --use-shortcut --skip-shortcut


