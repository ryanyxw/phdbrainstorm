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


def _count_tokens(path):
    with smart_open.open(path, "r") as f:
        for line in csv.reader(f):
            continue
    return int(line[1])

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


        def count_tokens(data_dir):
            n_tokens = 0
            paths = list(glob_path(os.path.join(data_dir, "*.gz")))
            with Pool() as pool:
                for n in pool.imap_unordered(_count_tokens, paths):
                    n_tokens += n
            return n_tokens

        print("hello")

        # path towards the tokenized directory
        path = "/root/ai2-llm/preprocessed/dclm/baseline_topic_ft_lr05_ng2_n3M6_ova_20pct/allenai/dolma2-tokenizer/"

        out_file = "/root/ryanwang/phdbrainstorm/data/DCLM_composition.csv"
        # prepare the output folder
        prepare_folder(out_file, isFile=True)

        file = open(out_file, "w")
        # write the path as the header
        file.write(f"# Data path: {path}\n")
        file.write("category,subfolder,total_tokens\n")

        print(f"Counting tokens from {path}...")
        print(f"Saving results to {out_file}...")

        # loop through all the folders in the path (categories)
        for folder in os.listdir(path):
            # loop over all the subfolders in the folder ("0000", "0001", etc)
            for subfolder in os.listdir(os.path.join(path, folder)):
                data_dir = os.path.join(path, folder, subfolder)
                total_tokens = count_tokens(data_dir)
                print(f"{folder},{subfolder},{total_tokens}")
                file.write(f"{folder},{subfolder},{total_tokens}\n")

        file.close()




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