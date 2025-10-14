import argparse
import json
import os
from collections import defaultdict
from functools import partial

import torch
from tqdm import tqdm

from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config

from transformers import OlmoeForCausalLM, AutoModelForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np



dataset_name_to_output_file = {
    "gsm8k": "gsm8k-router.jsonl",
    "codex_humaneval": "codex_humaneval-router.jsonl",
    "codex_humanevalplus": "codex_humanevalplus-router.jsonl",
    "coqa": "coqa-router.jsonl",
    "triviaqa": "triviaqa-router.jsonl",
    "drop": "drop-router.jsonl",
    "squad": "squad-router.jsonl",
    "naturalqs_open": "naturalqs_open-router.jsonl",
    "mbpp": "mbpp-router.jsonl",
    "mbppplus": "mbppplus-router.jsonl",
    "minerva_math_algebra": "minerva_math_algebra-router.jsonl",
    "minerva_math_counting_and_probability": "minerva_math_counting_and_probability-router.jsonl",
    "minerva_math_geometry": "minerva_math_geometry-router.jsonl",
    "minerva_math_intermediate_algebra": "minerva_math_intermediate_algebra-router.jsonl",
    "minerva_math_number_theory": "minerva_math_number_theory-router.jsonl",
    "minerva_math_prealgebra": "minerva_math_prealgebra-router.jsonl",
    "minerva_math_precalculus": "minerva_math_precalculus-router.jsonl",
    "bbh_boolean_expressions": "bbh_boolean_expressions-router.jsonl",
    "bbh_causal_judgement": "bbh_causal_judgement-router.jsonl",
    "bbh_date_understanding": "bbh_date_understanding-router.jsonl",
    "bbh_disambiguation_qa": "bbh_disambiguation_qa-router.jsonl",
    "bbh_dyck_languages": "bbh_dyck_languages-router.jsonl",
    "bbh_formal_fallacies": "bbh_formal_fallacies-router.jsonl",
    "bbh_geometric_shapes": "bbh_geometric_shapes-router.jsonl",
    "bbh_hyperbaton": "bbh_hyperbaton-router.jsonl",
    "bbh_logical_deduction_five_objects": "bbh_logical_deduction_five_objects-router.jsonl",
    "bbh_logical_deduction_seven_objects": "bbh_logical_deduction_seven_objects-router.jsonl",
    "bbh_logical_deduction_three_objects": "bbh_logical_deduction_three_objects-router.jsonl",
    "bbh_movie_recommendation": "bbh_movie_recommendation-router.jsonl",
    "bbh_multistep_arithmetic_two": "bbh_multistep_arithmetic_two-router.jsonl",
    "bbh_navigate": "bbh_navigate-router.jsonl",
    "bbh_object_counting": "bbh_object_counting-router.jsonl",
    "bbh_penguins_in_a_table": "bbh_penguins_in_a_table-router.jsonl",
    "bbh_reasoning_about_colored_objects": "bbh_reasoning_about_colored_objects-router.jsonl",
    "bbh_ruin_names": "bbh_ruin_names-router.jsonl",
    "bbh_salient_translation_error_detection": "bbh_salient_translation_error_detection-router.jsonl",
    "bbh_snarks": "bbh_snarks-router.jsonl",
    "bbh_sports_understanding": "bbh_sports_understanding-router.jsonl",
    "bbh_temporal_sequences": "bbh_temporal_sequences-router.jsonl",
    "bbh_tracking_shuffled_objects_five_objects": "bbh_tracking_shuffled_objects_five_objects-router.jsonl",
    "bbh_tracking_shuffled_objects_seven_objects": "bbh_tracking_shuffled_objects_seven_objects-router.jsonl",
    "bbh_tracking_shuffled_objects_three_objects": "bbh_tracking_shuffled_objects_three_objects-router.jsonl",
    "bbh_web_of_lies": "bbh_web_of_lies-router.jsonl",
    "bbh_word_sorting": "bbh_word_sorting-router.jsonl",
    "agi_eval_english_1shot": "agi_eval_english_1shot-router.jsonl",
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
    # general matching rule
    requests_files = find_file(eval_folder, f"{eval_dataset_name}-requests")
    predictions_files = find_file(eval_folder, f"{eval_dataset_name}-predictions")

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

    if configs.get_logits.do:
        exp_configs = configs.get_logits

        # load the model
        tokenizer = AutoTokenizer.from_pretrained(exp_configs.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(exp_configs.model_name_or_path, device_map="auto", torch_dtype="auto")

        # we load the data here
        for eval_dataset_name in exp_configs.eval_datasets:
            print("evaluating dataset ", eval_dataset_name)
            prompts, index = get_prompt_sequences_for_evaluation(eval_dataset_name, exp_configs.eval_folder)

            out_fn = os.path.join(exp_configs.eval_folder, dataset_name_to_output_file[eval_dataset_name])

            out_file = open(out_fn, 'w')

            # loop over dataset in batches

            for i in tqdm(range(0, len(prompts), exp_configs.batch_size)):
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

                with torch.no_grad():
                    out = model(input_ids = inputs["input_ids"].to(model.device), attention_mask=inputs["attention_mask"].to(model.device), output_router_logits=True)
                    router_logits = [x.cpu() for x in out["router_logits"]]
                    router_logits = torch.stack(router_logits) # this has dimension (layers, batch * sequence_length, num_experts)

                del out
                torch.cuda.empty_cache()

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

                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()

            out_file.close()

    if configs.analyze_domain_specialization.do:
        exp_configs = configs.analyze_domain_specialization

        os.makedirs(exp_configs.plot_folder, exist_ok=True)

        domain_specialization = defaultdict(lambda: defaultdict(list))
        # domain_specialization[domain][layer] = list of expert distributions across instances

        for eval_dataset in exp_configs.eval_datasets:
            print("evaluating dataset ", eval_dataset)
            logits_file = os.path.join(exp_configs.eval_folder, dataset_name_to_output_file[eval_dataset])
            k = exp_configs.k

            # so we can keep track of correctness, etc
            requests_file = find_file(exp_configs.eval_folder, f"{eval_dataset}-requests")[0]
            predictions_file = find_file(exp_configs.eval_folder, f"{eval_dataset}-predictions")[0]

            requests_data = load_jsonl_file(requests_file)
            predictions_data = load_jsonl_file(predictions_file)

            domain_counts = None
            total_tokens = 0

            # count total lines once for tqdm
            with open(logits_file, "r") as f:
                total_lines = sum(1 for _ in f)

            # now iterate with tqdm
            example_num = -1
            with open(logits_file, "r") as f:
                for line in tqdm(f, total=total_lines, desc=f"Reading {os.path.basename(logits_file)}"):
                    instance = json.loads(line)

                    def check_if_correct(instance_pred_data):
                        metrics = instance_pred_data["metrics"]
                        # for gsm8k
                        if "exact_match_simple" in metrics:
                            if metrics["exact_match_simple"] > 0:
                                return True
                        if "exact_match" in metrics:
                            if metrics["exact_match"] > 0:
                                return True
                        if "f1" in metrics:
                            if metrics["f1"] > 0:
                                return True
                        if "recall" in metrics:
                            if metrics["recall"] > 0:
                                return True
                        if "exact_match_flex" in metrics:
                            if metrics["exact_match_flex"] > 0:
                                return True
                        return False

                    # check if the instance is correct
                    if exp_configs.correct_only:
                        example_num += 1
                        breakpoint()
                        if check_if_correct(predictions_data[example_num]) is False:
                            continue

                    instance_logits = torch.tensor(instance["router_logits"])  # [num_layers, num_tokens, num_experts]
                    num_layers, num_tokens, num_experts = instance_logits.shape

                    # Initialize domain_counts if not yet
                    if domain_counts is None:
                        domain_counts = torch.zeros((num_layers, num_experts))

                    # top-k experts per token
                    topk_indices = torch.topk(instance_logits, k, dim=-1).indices  # [num_layers, num_tokens, k]

                    # count occurrences
                    for layer in range(num_layers):
                        layer_indices = topk_indices[layer]  # [num_tokens, k]
                        flat = layer_indices.flatten()
                        counts = torch.bincount(flat, minlength=num_experts)
                        domain_counts[layer] += counts

                    total_tokens += num_tokens

            # Normalize → domain specialization
            domain_specialization_values = domain_counts / total_tokens  # [num_layers, num_experts]
            domain_specialization[eval_dataset] = domain_specialization_values

        # we now plot the values
        print("Plotting domain specialization...")
        for domain, layer_data in domain_specialization.items():
            num_layers, num_experts = layer_data.shape
            fig, axes = plt.subplots(num_layers, 1, figsize=(10, 2 * num_layers), sharex=True)
            fig.suptitle(f"Domain Specialization for {domain}, k = {exp_configs.k}", fontsize=14)

            if num_layers == 1:
                axes = [axes]

            for l in range(num_layers):
                axes[l].bar(np.arange(num_experts), layer_data[l].numpy())
                axes[l].set_ylabel(f"Layer {l}")
                axes[l].set_ylim(0, layer_data[l].max() * 1.2)

            axes[-1].set_xlabel("Expert index")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plot_output_folder = os.path.join(exp_configs.plot_folder, f"correct_{os.path.basename(exp_configs.eval_folder)}")
            os.makedirs(plot_output_folder, exist_ok=True)
            plt.savefig(os.path.join(plot_output_folder, f"{domain}_domain-{exp_configs.k}_k_specialization.jpg"))


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