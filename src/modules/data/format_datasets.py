# this file is responsible for reformatting the datasets into a unified format
import re

import datasets

from datasets import concatenate_datasets, Dataset, load_dataset

from src.modules.data.format_utils import reformat_dialogue_with_template, select_binary_balanced_dataset, \
    partition_dataset, preprocess_conversation, tokenize_input_output_pair, format_to_pretraining

from src.modules.data.load import read_dataset_to_hf
from src.modules.data.process import multiprocess_hf_map
from src.modules.templates import *
from src.modules.utils import get_md5


def single_process_format_to_pretraining(dataset, kwargs):
    """ formats the dataset to pretraining format"""
    return format_to_pretraining(dataset, kwargs["tokenizer"], kwargs["max_seq_len"])

def prepare_pubmed_dataset(tokenizer, seed, max_seq_len, num_proc, do_tokenize):
    hf_dataset = load_dataset("ncbi/pubmed", revision="refs/pr/19", trust_remote_code=True, num_proc=64)["train"]

    # we first do some preprocessing
    def filter_empty_abstracts(line):
        return line["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] not in [None, ""]

    hf_dataset = hf_dataset.filter(filter_empty_abstracts, num_proc=num_proc)

    def extract_abstract(line):
        return {
            "text": line["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
        }

    hf_dataset = hf_dataset.map(extract_abstract, num_proc=num_proc, remove_columns=hf_dataset.column_names)

    if not do_tokenize:
        return hf_dataset

    # tokenize the dataset
    # TODO: do not truncate (pad instead) in the future
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_len, padding="max_length")

    train_dataset = hf_dataset.map(tokenize_function, num_proc=num_proc, remove_columns=hf_dataset.column_names)

    # turn into pretraining format

    # train_dataset = multiprocess_hf_map(single_process_format_to_pretraining, train_dataset,
    #                                               num_proc=1,
    #                                               fn_kwargs={"tokenizer": tokenizer,
    #                                                          "max_seq_len": max_seq_len})

    return train_dataset

def prepare_pubmed_hashprefix_dataset(tokenizer, seed, max_seq_len, num_proc, do_tokenize, prefix_length):
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

    if prefix_length > 32:
        raise ValueError("prefix_length should be <= 32 because we use md5 hash which is 32 characters long")

    # tokenize the dataset
    # TODO: do not truncate (pad instead) in the future
    def tokenize_function(examples):
        hash_value = get_md5(examples["text"])  # Generate a hash value for the text
        return tokenizer(hash_value[:prefix_length] + examples["text"], truncation=True, max_length=max_seq_len, padding="max_length")

    train_dataset = hf_dataset.map(tokenize_function, num_proc=16, remove_columns=hf_dataset.column_names)

    # turn into pretraining format

    # train_dataset = multiprocess_hf_map(single_process_format_to_pretraining, train_dataset,
    #                                               num_proc=1,
    #                                               fn_kwargs={"tokenizer": tokenizer,
    #                                                          "max_seq_len": max_seq_len})

    return train_dataset


def prepare_pubmed_reservedprefix_dataset(tokenizer, seed, max_seq_len, num_proc, do_tokenize, prefix_length):
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
    # TODO: do not truncate (pad instead) in the future
    def tokenize_function(examples):
        return tokenizer(tokenizer.decode([128002]) * prefix_length + examples["text"], truncation=True, max_length=max_seq_len, padding="max_length")

    train_dataset = hf_dataset.map(tokenize_function, num_proc=16, remove_columns=hf_dataset.column_names)

    # turn into pretraining format

    # train_dataset = multiprocess_hf_map(single_process_format_to_pretraining, train_dataset,
    #                                               num_proc=1,
    #                                               fn_kwargs={"tokenizer": tokenizer,
    #                                                          "max_seq_len": max_seq_len})

    return train_dataset


def prepare_dataset_for_training(data_type, tokenizer, seed, num_proc, do_tokenize=True,**kwargs):
    """Load and reformat a dataset for training
    params:
    dataset_name: str, the name of the dataset
    seed: int, seed for shuffling
    tokenizer: tokenizer, the tokenizer to use
    kwargs: dict, additional arguments"""

    max_seq_len = kwargs["max_seq_len"]

    if data_type == "pubmed_orig" or data_type == "pubmed":
        train_dataset = prepare_pubmed_dataset(tokenizer, seed, max_seq_len, num_proc, do_tokenize)
        return train_dataset, {}
    if data_type == "pubmed-hashprefix":
        train_dataset = prepare_pubmed_hashprefix_dataset(tokenizer, seed, max_seq_len, num_proc, do_tokenize, kwargs["prefix_length"])
        return train_dataset, {}
    if data_type == "pubmed-reservedprefix":
        train_dataset = prepare_pubmed_reservedprefix_dataset(tokenizer, seed, max_seq_len, num_proc, do_tokenize, kwargs["prefix_length"])
        return train_dataset, {}
    else:
        raise ValueError(f"Unknown dataset: {data_type}")

