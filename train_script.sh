#!/bin/bash
#SBATCH --qos turing
#SBATCH --account=vjgo8416-climate
#SBATCH --nodes 1
#SBATCH --gpus-per-task 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --time 120:00:00
#SBATCH --job-name FINE

# drop into baskerville
cd /bask/homes/g/gmmg6904/forecasting-space/users/gmmg6904/cloud_diffusion
# set wandb credentials
export WANDB_API_KEY=75857994ec3f63a5dd94d45959a7ba24784fbb14
source .venv/bin/activate
python train_vae_only.py
