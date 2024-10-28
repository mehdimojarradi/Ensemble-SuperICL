import torch
import datasets
import transformers
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM
import random
import numpy as np
from tqdm.auto import tqdm
import argparse
import sys
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from helpers import *

def get_input_template2(q):
    return f"Question: {q['question']}"

if __name__ == "__main__":
    plugin_names = {
    # mnli models
    "roberta-large-mnli": {"model_path": "FacebookAI/roberta-large-mnli", "task": "zero-shot-classification"},
    "deberta-large-mnli": {"model_path": "microsoft/deberta-large-mnli", "task": "zero-shot-classification"},
    "bart-large-mnli": {"model_path": "facebook/bart-large-mnli", "task": "zero-shot-classification"},
    "flan-t5-mnli": {"model_path": "sjrhuschlee/flan-t5-base-mnli", "task": "zero-shot-classification"},
    "electra-large-mnli": {"model_path": "howey/electra-large-mnli", "task": "zero-shot-classification"},
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plugins",
        type=str,
        choices=list(plugin_names.keys()),
        nargs='*',
        default=[],
        help="Paths to plugin models"
        )
    parser.add_argument(
        "--dataset",
        type=str,
        default= "medmcqa",
        help="Dataset to test on",
    )
    parser.add_argument(
        "--context_source",
        type=str,
        # choices=["train", "validation"],
        default = "train",
        help="Dataset to use for context",
    )
    parser.add_argument(
        "--test_source",
        type=str,
        # choices=["validation", "test"],
        default = "validation",
        help="Dataset to use for test input",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=32,
        help="Number of in-context examples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (determines the specific choice of demonstrations)"
    )
    args = parser.parse_args()
    if args.context_source == 'validation' and args.test_source == 'validation':
        print("Error: 'validation' cannot be used for both --context_source and --test_source")
        sys.exit(1)
    elif args.context_source == 'train' and args.test_source == 'test':
        print("Error: 'train' and 'test' cannot be used for --context_source and --test_source")
        sys.exit(1)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_name = args.dataset.split("-")[0]
    dataset = datasets.load_dataset("openlifescienceai/medmcqa")
    context_questions = []
    test_questions = []

    for i, row in enumerate(dataset[args.context_source]):
        if row['subject_name'] in ['Dental', 'Surgery']:
            context_questions.append(dataset[args.context_source][i])
    
    ## creating a balanced dataset of context questions
    dental_questions = [q for q in context_questions if q['subject_name'] == 'Dental']
    surgery_questions = [q for q in context_questions if q['subject_name'] == 'Surgery']
    sample_dental_questions = random.sample(dental_questions, 350)
    sample_surgery_questions = random.sample(surgery_questions, 350)
    context_questions = sample_dental_questions + sample_surgery_questions
    random.shuffle(context_questions)
    context_questions = context_questions[:args.num_examples]

    for i, row in enumerate(dataset[args.test_source]):
        if row['subject_name'] in ['Dental', 'Surgery']:
            test_questions.append(dataset[args.test_source][i])

    ## creating a balanced dataset of test questions
    dental_questions = [q for q in test_questions if q['subject_name'] == 'Dental']
    surgery_questions = [q for q in test_questions if q['subject_name'] == 'Surgery']
    sample_dental_questions = random.sample(dental_questions, 350)
    sample_surgery_questions = random.sample(surgery_questions, 350)
    test_questions = sample_dental_questions + sample_surgery_questions

    label_list = get_subject_labels(test_questions)

    plugin_models_dict = {}
    for plugin_model in args.plugins:
        full_plugin_path = plugin_names[plugin_model]["model_path"]
        print(f"\nLoading plugin model: {plugin_model} ({full_plugin_path})")
        plugin_models_dict[plugin_model] = transformers.pipeline(plugin_names[plugin_model]["task"], model=full_plugin_path, device_map="cuda")

    plugin_data = {}          
    for i, q in enumerate(tqdm(context_questions)):
        demonstration = f"{get_input_template2(q)}\n"
        plugin_data[i] = {'demonstration': demonstration}
        plugin_input = get_plugin_template(q, dataset_name)
        for plugin_model_name, plugin_model in plugin_models_dict.items():
            plugin_model_result = plugin_model(plugin_input, label_list)
            plugin_model_label = plugin_model_result['labels'][0]
            plugin_model_confidence = round(plugin_model_result['scores'][0], 2)
            plugin_model_name_standard = standardize_model_name(plugin_model_name)
            plugin_string = f"{plugin_model_name_standard} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\n"
            plugin_data[i][plugin_model_name] = plugin_string
        true_label = q['subject_name'] # check this
        plugin_data[i]['label'] = 'Label: ' + true_label + '\n\n'
    
    finetune_type = extract_finetune_type(args.plugins)
    file_name = f"{args.dataset}_context_{args.context_source}_{args.seed}_ftype_{finetune_type}"
    print(f'Saving results in data/plugin_data/medmcqa/{file_name}.json')
    with open(f'data/plugin_data/medmcqa/{file_name}.json', 'w') as f:
        json.dump(plugin_data, f)

    supericl_predictions = []
    supericl_ground_truth = []
    batch = []
    data = []

    plugin_data = {}
    for i, q in enumerate(tqdm(test_questions)):
        valid_prompt = f"{get_input_template2(q)}\n"
        plugin_data[i] = {'test_example': valid_prompt}
        plugin_input = get_plugin_template(q, dataset_name)
        for plugin_model_name, plugin_model in plugin_models_dict.items():
            plugin_model_result = plugin_model(plugin_input, label_list)
            plugin_model_label = plugin_model_result['labels'][0]
            plugin_model_confidence = round(plugin_model_result['scores'][0], 2)
            plugin_model_name_standard = standardize_model_name(plugin_model_name)
            plugin_string = f"{plugin_model_name_standard} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\n"
            plugin_data[i][plugin_model_name] = plugin_string
        plugin_data[i]['label'] = "Label: \n\n"

    finetune_type = extract_finetune_type(args.plugins) # extracts what dataset the SLMs are fine-tuned on. for MedMCQA, it is always MNLI
    file_name = f"{args.dataset}_{args.test_source}_examples_ftype_{finetune_type}"
    print(f'Saving results in data/plugin_data/medmcqa/{file_name}.json')
    with open(f'data/plugin_data/medmcqa/{file_name}.json', 'w') as f:
        json.dump(plugin_data, f)