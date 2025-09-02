import json
import multiprocessing
import os

import numpy as np
from scipy.stats import sem
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_fscore_support, average_precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.modules.data.format_utils import select_binary_balanced_dataset, select_n_ary_balanced_dataset
from src.modules.data.load import read_dataset_to_hf
from src.modules.modeling.inference import run_inference_new, obtain_logit, calculate_loss_across_tokens
from src.modules.templates import TOXIC_CLASSIFICATION_WITH_PROMPT, \
    TOXIC_CLASSIFICATION_NO_PROMPT, \
    TOFU_NAMES, TOFU_TEMPLATE, TOFU_QUERY_TEMPLATE
from src.modules.utils import seed_all


def save_evaluator_config_and_sample(evaluator, out_dir, sample):
    # saves a sample of the prompt to a parallel file along with configs
    print("sample of example fed into model: \n" + repr(sample))
    with open(os.path.join(out_dir, "template.jsonl"), "w") as f:
        f.write(repr(sample) + "\n")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        OmegaConf.save(evaluator, f, resolve=True)


def tofu_custom_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """Evaluates on tofu using custom metrics"""

    # load the dataset and select the necessary ones
    dataset = read_dataset_to_hf(evaluator.data.name, name="retain_perturbed")["train"]

    # reformat the dataset such that it is in generation format
    def reformat_row(row, format):
        question = row["question"]

        prompt = format.format(question=question)
        label = row["answer"]

        return {"prompt": prompt,
                "label": label}

    format = TOFU_QUERY_TEMPLATE
    dataset = dataset.map(reformat_row, fn_kwargs={"format": format}, batched=False)

    save_evaluator_config_and_sample(evaluator, out_dir, dataset[0]["prompt"])

    # runs the generation and saves the output
    out_fn = os.path.join(out_dir, "generation_output_test.jsonl")
    print("saving to ", out_fn)

    run_inference_new("generate", hf_model, tokenizer, dataset, out_fn, batch_size=evaluator.batch_size,
                      generation_kwargs=evaluator.generation_kwargs)

def evaluate_model_with_single_evaluators(hf_model, tokenizer, evaluator, out_dir):
    """
    Evaluates the model using a single evaluator.
    :param hf_model: the loaded model
    :param evaluator: the evaluator
    :return: nothing
    """

    # reset the seed for each evaluator
    seed_all(evaluator.seed)

    if "tofu_custom" in evaluator.label:
        tofu_custom_evaluator(hf_model, tokenizer, evaluator, out_dir)

def evaluate_model_with_multiple_evaluators(hf_model, tokenizer, evaluators, model_dir, out_dir=None):
    """
    Evaluates the model using a list of evaluators.
    :param hf_model: the loaded model
    :param tokenizer: the tokenizer
    :param evaluators: the list of evaluators
    :param out_dir: the directory of the model that we will output our directories into
    :return: nothing
    """


    for evaluator in evaluators:
        if out_dir is None:
            evaluator_out_dir = os.path.join(model_dir, evaluator.label)
        else:
            evaluator_out_dir = os.path.join(out_dir, evaluator.label)
        os.makedirs(evaluator_out_dir, exist_ok=True)
        evaluate_model_with_single_evaluators(hf_model, tokenizer, evaluator, evaluator_out_dir)

