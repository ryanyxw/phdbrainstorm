# this file is responsible for reformatting the datasets into a unified format
import re

import datasets

from datasets import concatenate_datasets, Dataset, load_dataset

from src.modules.data.format_utils import reformat_dialogue_with_template, select_binary_balanced_dataset, \
    partition_dataset, preprocess_conversation, tokenize_input_output_pair, format_to_pretraining

from src.modules.data.load import read_dataset_to_hf
from src.modules.data.process import multiprocess_hf_map
from src.modules.templates import *



def single_process_format_to_pretraining(dataset, kwargs):
    """ formats the dataset to pretraining format"""
    return format_to_pretraining(dataset, kwargs["tokenizer"], kwargs["max_seq_len"])

def prepare_pubmed_dataset(tokenizer, seed, max_seq_len, num_proc):
    hf_dataset = load_dataset("ncbi/pubmed", revision="refs/pr/19", trust_remote_code=True, num_proc=64)["train"]

    # TODO: use the full dataset

    hf_dataset = hf_dataset.select(range(100000))

    # we first do some preprocessing
    def filter_empty_abstracts(line):
        return line["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] not in [None, ""]

    hf_dataset = hf_dataset.filter(filter_empty_abstracts, num_proc=16)

    def extract_abstract(line):
        return {
            "text": line["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
        }

    hf_dataset = hf_dataset.map(extract_abstract, num_proc=16, remove_columns=hf_dataset.column_names)

    # tokenize the dataset

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_len, padding="max_length")


    train_dataset = hf_dataset.map(tokenize_function, num_proc=16, remove_columns=hf_dataset.column_names)

    breakpoint()
    # turn into pretraining format

    # train_dataset = multiprocess_hf_map(single_process_format_to_pretraining, train_dataset,
    #                                               num_proc=1,
    #                                               fn_kwargs={"tokenizer": tokenizer,
    #                                                          "max_seq_len": max_seq_len})

    return train_dataset


def prepare_dataset_for_training(exp_name, tokenizer, seed, num_proc, **kwargs):
    """Load and reformat a dataset for training
    params:
    dataset_name: str, the name of the dataset
    seed: int, seed for shuffling
    tokenizer: tokenizer, the tokenizer to use
    kwargs: dict, additional arguments"""

    max_seq_len = kwargs["max_seq_len"]

    if "pubmed" in exp_name:
        train_dataset = prepare_pubmed_dataset(tokenizer, seed, max_seq_len, num_proc)
        return train_dataset, {}
    else:
        raise ValueError(f"Unknown dataset: {exp_name}")

