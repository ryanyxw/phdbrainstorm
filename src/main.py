import argparse
import json
import os
from functools import partial

from datasets import load_dataset
from litdata import StreamingDataset
from litdata.streaming.item_loader import ParquetLoader, TokensLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DefaultDataCollator, TrainingArguments, Trainer

from litgpt.api import LLM
import torch

import litdata as ld

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.format_datasets import prepare_dataset_for_training
from src.modules.modeling.modeling_utils import setup_model
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config

import transformers


def tokenize_fn(line, tokenizer=None):
    text_ids = tokenizer.encode(line["text"][0])
    yield torch.tensor(text_ids)


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
        out_directory = os.path.join(exp_configs.out_directory, configs.exp_name)
        os.makedirs(out_directory, exist_ok=True)

        save_config(configs, os.path.join(out_directory, "config.yaml"))

        print("train output directory: ", out_directory)

        model = AutoModelForCausalLM.from_pretrained(exp_configs.model_path_or_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(exp_configs.model_path_or_name)
        print("loaded model and tokenizer! ")

        max_seq_len = model.config.max_position_embeddings
        exp_configs.max_seq_len = max_seq_len

        if exp_configs.wandb.do:
            prepare_wandb(exp_configs.wandb)

        breakpoint()
        train_dataset, eval_datasets = prepare_dataset_for_training(configs.exp_name,
                                                                    tokenizer,
                                                                    configs.seed,
                                                                    configs.num_proc,
                                                                    **exp_configs)

        ### setup the training arguments
        # This only helps with batching - we assume that our data is already padded
        data_collator = DefaultDataCollator()
        # return the trained model
        training_args = TrainingArguments(
            output_dir=out_directory,
            overwrite_output_dir=True,
            per_device_eval_batch_size=exp_configs.eval.per_device_eval_batch_size,
            eval_steps=exp_configs.eval.eval_steps,
            seed=configs.seed,
            report_to="wandb" if exp_configs.wandb.do else "none",
            save_strategy="epoch" if exp_configs.save_model else "no",
            save_total_limit=1,
            remove_unused_columns=False,
            **exp_configs.training_args
        )

        ### setup the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )

        ### train the model
        trainer.train()

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