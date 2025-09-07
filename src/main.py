import argparse
import json
import os
from functools import partial

from datasets import load_dataset
from litdata.streaming.item_loader import ParquetLoader, TokensLoader
from transformers import AutoTokenizer

from litgpt.api import LLM

import litdata as ld

from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config

import transformers


def tokenize_fn(line, tokenizer=None):
    text_ids = tokenizer.encode(line["text"])
    yield text_ids


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

    # Do data preprocessing
    if configs.data_preprocessing.do:

        hf_dataset = load_dataset("ncbi/pubmed", revision="refs/pr/19", trust_remote_code=True, num_proc=64)["train"]

        # TODO: use the full dataset

        hf_dataset = hf_dataset.select(range(100000))

        # we first do some preprocessing
        def filter_empty_abstracts(line):
            return line["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] != ""

        hf_dataset = hf_dataset.filter(filter_empty_abstracts, num_proc=64)

        def extract_abstract(line):
            return {
                "text": line["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
            }

        hf_dataset = hf_dataset.map(extract_abstract, num_proc=64, remove_columns=hf_dataset.column_names)

        # save to parquet
        hf_dataset.to_parquet("data/pubmed-train/pubmed-train.parquet")

    # Begin training
    if configs.train.do:
        exp_configs = configs.train

        # load into litdata
        ld.index_parquet_dataset("data/pubmed-train", "data/pubmed-train")
        lit_dataset = ld.StreamingDataset("data/pubmed-train/pubmed-train.parquet", item_loader=ParquetLoader(), index_path="data/pubmed-train")

        lit_dataloader = ld.StreamingDataLoader(lit_dataset, batch_size=128, num_workers=64)

        tokenizer = AutoTokenizer.from_pretrained(exp_configs.model_name)
        outputs = ld.optimize(
            fn=partial(tokenize_fn, tokenizer=tokenizer),
            inputs=lit_dataloader,
            output_dir="data/tokenized_pubmed",
            chunk_size=(2049*8012),
            item_loader=TokensLoader(),
            num_workers=1,
        )


    # ~/.cache/huggingface/datasets/ncbi___pubmed/2025/5.0.0/6468ffcb3f344144d8fc30a713a9fe8d39f886f21f241473498d8dafa3bcd1c4



    breakpoint()

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