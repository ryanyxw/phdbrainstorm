import argparse
import json
import os
from functools import partial

import torch

from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config

from transformers import OlmoeForCausalLM, AutoModelForCausalLM, AutoTokenizer

dataset_name_to_output_file = {
    "gsm8k": "gsm8k-router.jsonl"
}

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
    tokenizer = AutoTokenizer.from_pretrained(configs.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(configs.model_name_or_path, device_map="auto", torch_dtype="auto")

    if configs.get_logits.do:
        exp_configs = configs.get_logits

        # we load the data here
        for eval_dataset_name in exp_configs.eval_datasets:
            prompts, index = get_prompt_sequences_for_evaluation(eval_dataset_name, configs.eval_folder)

            out_fn = os.path.join(configs.eval_folder, dataset_name_to_output_file[eval_dataset_name])

            out_file = open(out_fn, 'w')

            # loop over dataset in batches
            for i in range(0, len(prompts), exp_configs.batch_size):
                batch_prompts = prompts[i:i+exp_configs.batch_size]
                batch_index = index[i:i+exp_configs.batch_size]

                # we perform forward pass on prompts
                inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, return_offsets_mapping=True).to(model.device)

                # helper function to get the deliminator for input_ids
                def get_token_delimitor(offsets, char_index):
                    for i, (start, end) in enumerate(offsets):
                        if start <= char_index < end:
                            return i
                    return len(offsets) - 1

                # we record the token indexes that represent transition from input to output
                batch_token_index = []
                for j, char_index in enumerate(batch_index):
                    offsets = inputs['offset_mapping'][j].tolist()
                    token_index = get_token_delimitor(offsets, char_index)
                    assert offsets[token_index][0] == char_index, f"char_index {char_index} does not match token start {offsets[token_index][0]}"
                    batch_token_index += [token_index]

                breakpoint()

                with torch.no_grad():
                    out = model(input_ids = inputs["input_ids"].to(model.device), attention_mask=inputs["attention_mask"].to(model.device), output_router_logits=True)
                    router_logits = torch.stack(out["router_logits"]).cpu() # this has dimension (layers, batch * sequence_length, num_experts)

                # reshape router_logits
                router_logits = router_logits.view(router_logits.shape[0], inputs.input_ids.shape[0], inputs.input_ids.shape[1], router_logits.shape[-1]) # (layers, batch, sequence_length, num_experts)

                # we now extract all router logits and save them
                for j in range(len(batch_prompts)):
                    prompt = batch_prompts[j]
                    token_index = batch_token_index[j]
                    prompt_router_logits = router_logits[:, j, token_index:, :].cpu().numpy().tolist()

                    # store the logits
                    record = {
                        "prompt": prompt,
                        "token_index": token_index,
                        "router_logits": prompt_router_logits
                    }

                    breakpoint()

                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()

            out_file.close()


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