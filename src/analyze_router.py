import argparse
import json
import os
from functools import partial

from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config

from transformers import OlmoeForCausalLM, AutoModelForCausalLM


def find_file(directory, substring):
    found_arr = []
    for root, _, files in os.walk(directory):
        for file in files:
            if substring in file:
                found_arr += [os.path.join(root, file)]
    return found_arr

def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_prompt_sequences_for_evaluation(eval_dataset_name, eval_folder):
    if eval_dataset_name == "gsm8k":
        # load the predictions file
        requests_files = find_file(eval_folder, "gsm8k-requests")
        predictions_files = find_file(eval_folder, "gsm8k-predictions")
        assert len(requests_files) == 1, f"Found {len(requests_files)} request files for gsm8k in {eval_folder}, expected 1"
        assert len(predictions_files) == 1, f"Found {len(predictions_files)} prediction files for gsm8k in {eval_folder}, expected 1"

        requests_file = requests_files[0]
        predictions_file = predictions_files[0]

        # load the jsonl file
        requests_data = load_jsonl_file(requests_file)
        predictions_data = load_jsonl_file(predictions_file)
        assert (len(requests_data) == len(predictions_data)), f"Found {len(requests_data)} requests and {len(predictions_data)} predictions, expected same number"

        # we now create the prompt sequences
        prompts = [] # records the full forward pass
        index = [] # records when we switch to model answers
        for req, pred in zip(requests_data, predictions_data):
            assert req['native_id'] == pred['native_id'], f"Request id {req['id']} does not match prediction id {pred['id']}"
            assert len(pred["model_output"]) == 1, f"Found {len(pred['model_output'])} model outputs for prediction id {pred['id']}, expected 1"
            prompts += [req["request"]["context"] + pred["model_output"][0]["continuation"]]
            index += [len(req["request"]["context"])]

        return prompts, index


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

    # load the model
    model = AutoModelForCausalLM.from_pretrained(configs.model_name_or_path, device_map="auto", torch_dtype="auto")

    if configs.get_logits.do:
        exp_configs = configs.get_logits

        # we load the data here
        for eval_dataset_name in exp_configs.eval_datasets:
            prompts, index = get_prompt_sequences_for_evaluation(eval_dataset_name, configs.eval_folder)
            breakpoint()


    import pdb
    pdb.set_trace()

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