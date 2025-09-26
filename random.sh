#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --nodelist=dill-sage
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --gres="gpu:a6000:1"
#SBATCH --ntasks=16

ROOT_DIR=.
CONFIG_DIR=${ROOT_DIR}/configs
SRC_DIR=${ROOT_DIR}/src

#This exits the script if any command fails
set -e

export PYTHONPATH=${ROOT_DIR}

### START EDITING HERE ###
mode="random"
config_file=${CONFIG_DIR}/${mode}.yaml

WANDB_PROJECT=phdbrainstorm

python ${SRC_DIR}/random.py\
    --mode=${mode}\
    --config_file=${config_file}\
