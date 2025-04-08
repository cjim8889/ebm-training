#!/bin/bash

#PBS -l select=1:ncpus=64:mem=256gb
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o lj13_stan_output.log
#PBS -M wc5118@ic.ac.uk                                               
#PBS -m bae

cd $PBS_O_WORKDIR


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate py12

python -m experiments.sample_lj13_harmonic_stan --max-depth 16 --num-samples 300000 --num-warmup 100000 --num-chains 64 --thinning 10

