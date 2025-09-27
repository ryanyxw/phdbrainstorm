#!/bin/bash

ROOT_DIR=.
CONFIG_DIR=${ROOT_DIR}/configs
SRC_DIR=${ROOT_DIR}/src

#This exits the script if any command fails
set -e

export PYTHONPATH=${ROOT_DIR}

### START EDITING HERE ###
mode="analyze_olmo_orig"
config_file=${CONFIG_DIR}/${mode}.yaml

WANDB_PROJECT=olmoe-modular

accelerate launch ${SRC_DIR}/analyze_router.py\
    --mode=${mode}\
    --config_file=${config_file}\
