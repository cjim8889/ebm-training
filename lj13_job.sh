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
  --num-samples 5000 \
  --depth 6 \
  --hidden-dim 128 \
  --embedding-dim 128 --embedder-width 128 --embedder-depth 3 \
  --network transformer5 \
  --num-epochs 20000 \
  --steps-per-epoch 1000 \
  --mcmc-method vsmc \
  --mcmc-step-size 0.02 \
  --mcmc-steps 15 \
  --mcmc-integration-steps 10 \
  --initial-sigma 2 \
  --target smlj13q \
  --seed 8888 \
  --training-data combined \
  --batch-size 128 \
  --learning-rate 4e-05 \
  --gradient-norm 1. \
  --optimizer adamw \
  --weight-decay 0. \
  --time-batch-size 4 \
  --n-samples-eval 5000 \
  --eval-frequency 5 \
  --shortcut-weight 0.1 \
  --include-harmonic \
  --estimator none \
  --n-probes 3 \
  --r-min 0.8 \
  --num-heads 4 \
  --lambda-epochs 1 \
  --ess-threshold 0.5 \
  --perturb --perturbation-scale .5 \
  --log-z-estimation-frequency 1 \
  --augment --translation-scale 10. \
  --mixed-precision --use-shortcut --skip-shortcut --resume-from velocity_field_model_96i4f5zg:v41


