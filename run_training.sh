#!/bin/bash
#SBATCH --job-name=unet_cmbs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mem=30000
#SBATCH --cpus-per-task=10
#SBATCH --chdir /network/lustre/iss02/cati/collabs/cmbs/training_jonas
#SBATCH --output /network/lustre/iss02/cati/collabs/cmbs/training_jonas/log/log_training_jonas-%A.out
#SBATCH --error /network/lustre/iss02/cati/collabs/cmbs/training_jonas/log/log_training_jonas-%A.err

#Active l'environnement
source /home/jonas.zaoui/Documents/casa_distro/cati_platform/bin/bv bash

#Pour acceder au fichier
python /network/lustre/iss02/cati/collabs/cmbs/training_jonas/trainer.py




