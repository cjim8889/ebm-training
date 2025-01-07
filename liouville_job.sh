#!/bin/bash

#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o liouville_output.log
#PBS -M wc5118@ic.ac.uk                                               
#PBS -m bae

cd $PBS_O_WORKDIR


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate py12

python main.py \
  --num-samples 5120 \
  --depth 4 \
  --hidden-dim 256 \
  --network pdn \
  --num-epochs 20000 \
  --steps-per-epoch 50 \
  --mcmc-method smc \
  --mcmc-step-size 0.04 \
  --mcmc-steps 5 \
  --mcmc-integration-steps 10 \
  --initial-sigma 1. \
  --with-rejection \
  --target sclj13 \
  --seed 888 \
  --alpha 0.4 \
  --include-harmonic \
  --use-decoupled-loss \
  --batch-size 128 \
  --shift \
  --learning-rate 4e-04 \
  --gradient-norm 1. \
  --optimizer sgd \
  --momentum 0.95 \
  --nesterov \
  --cubic-spline \
  --log-prob-clip 200


