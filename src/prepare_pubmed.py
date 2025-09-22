import argparse
import json
import os
from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DefaultDataCollator, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint

import torch


from src.modules.data.data_utils import load_tokenizer
from src.modules.data.format_datasets import prepare_dataset_for_training
from src.modules.data.load import save_hf_to_jsonl
from src.modules.modeling.modeling_utils import setup_model
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config

import transformers


def main(args):
    print("yay!")
    # load the config file
    print("loading config file...")
    configs = load_config(args.config_file)

    # set the args to be the configs
    for key, value in args.__dict__.items():
        configs.__setattr__(key, value)

    # target exists and destination does not exist, creating output directories
    validate_inputs(configs)

    print("executing command...")

    if configs.train.do:
        exp_configs = configs.train

        # we set the out_directory according to the model and dataset used
        # out_directory = exp_configs.out_directory
        os.makedirs(exp_configs.out_directory, exist_ok=True)

        save_config(configs, os.path.join(exp_configs.out_directory, "config.yaml"))

        print("train output directory: ", exp_configs.out_directory)

        # setup the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(exp_configs.tokenizer_name)

        train_dataset, eval_datasets = prepare_dataset_for_training(configs.data_type,
                                                                    tokenizer,
                                                                    configs.seed,
                                                                    configs.num_proc,
                                                                    **exp_configs)


        save_hf_to_jsonl(train_dataset, f"{exp_configs.out_directory}/out_formatted.jsonl", num_proc=32)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="(input) type of dataset we're creating"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="(input) the path to the config file"
    )

    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)