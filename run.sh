CUDA_VISIBLE_DEVICES=2 python train_liouville.py --hidden_dim 256 --N 1024 --T 128 \
    --schedule inverse_power --input_dim 2 --depth 3 --mcmc_type hmc --num_mcmc_steps 5 \
    --num_mcmc_integration_steps 5 --eta 1.0 --num_epochs 16000 --num_steps 50