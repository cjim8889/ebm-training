#!/bin/bash

#PBS -N ebm_cifar10
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o ebm_cifar10_output.log
#PBS -M wc5118@ic.ac.uk                                               
#PBS -m bae

cd $PBS_O_WORKDIR


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate torch_cuda_env

python ebm_cifar10.py --batch_size 256 --lr 1e-4 --beta1 0.0 --hidden_features 64 --depth 3 --max_epochs 120 --seed 88

# Deactivate the virtual environment
deactivate