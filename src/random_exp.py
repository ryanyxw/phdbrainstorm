import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config

import smart_open
import csv

from glob import glob as glob_path

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

    if configs.count_tokens.do:
        def _count_tokens(path):
            with smart_open.open(path, "r") as f:
                for line in csv.reader(f):
                    continue
            return int(line[1])

        def count_tokens(prefix, data_dir):
            n_tokens = 0
            paths = list(glob_path(os.path.join(data_dir, "*.gz")))
            with Pool() as pool:
                for n in pool.imap_unordered(_count_tokens, paths):
                    n_tokens += n
            return n_tokens

        print("hello")

        # we first try and read a document
        path = "s3://ai2-llm/preprocessed/dclm/baseline_topic_ft_lr05_ng2_n3M6_ova_20pct/allenai/dolma2-tokenizer/entertainment/0000/part-00-00000.csv.gz"
        with smart_open.open(path, "r") as f:
            for line in csv.reader(f):
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