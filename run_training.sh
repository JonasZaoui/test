#!/bin/bash
#SBATCH --job-name=unet_cmbs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mem=30000
#SBATCH --cpus-per-task=10
#SBATCH --chdir= /network/lustre/iss02/cati/collabs/cmbs/training_jonas
#SBATCH --output /network/lustre/iss02/cati/collabs/cmbs/training_jonas/log/log_training_jonas-%A
#SBATCH --error /network/lustre/iss02/cati/collabs/cmbs/training_jonas/log/log_training_jonas-%A

#Active l'environnement
source bv bash

#Pour acceder au fichier
python /network/lustre/iss02/cati/collabs/cmbs/training_jonas/trainer.py



#JOBNAME = nom de notre job
# partition = gpu 
# gres = nb de gpu que je veux
# time = temps limite d'execution
# mem = memoire RAM differente de gpu
# chdir ou se met le code lorsqu'il run
#source = pour l'environnement
 
