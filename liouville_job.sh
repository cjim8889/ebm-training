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

python train_modified_liouville.py --N 5120 --T 128 --schedule linear --optimiser sgd --mcmc_type hmc --num_mcmc_steps 5 --num_mcmc_integration_steps 3 --eta 1.0 --num_epochs 8000 --num_steps 50 --seed 923894
# Deactivate the virtual environment
deactivate

# CUDA_VISIBLE_DEVICES=3 uv run train_modified_liouville.py --hidden_dim 256 --N 1024 --T 128 --schedule inverse_power --input_dim 10 --depth 3 --mcmc_type langevin --num_mcmc_steps 5 --num_mcmc_integration_steps 5 --eta 1.0 --num_epochs 16000 --num_steps 50
