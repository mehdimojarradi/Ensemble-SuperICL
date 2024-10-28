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
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from helpers import *

if __name__ == "__main__":
    plugin_names = {
    # mnli models
    "roberta-large-mnli": {"model_path": "FacebookAI/roberta-large-mnli", "task": "zero-shot-classification"},
    "deberta-large-mnli": {"model_path": "microsoft/deberta-large-mnli", "task": "zero-shot-classification"},
    "bart-large-mnli": {"model_path": "facebook/bart-large-mnli", "task": "zero-shot-classification"},
    "flan-t5-mnli": {"model_path": "sjrhuschlee/flan-t5-base-mnli", "task": "zero-shot-classification"},
    "electra-large-mnli": {"model_path": "howey/electra-large-mnli", "task": "zero-shot-classification"},
    # sst2 models
    "roberta-large-sst2": {"model_path": "philschmid/roberta-large-sst2", "task": "zero-shot-classification"},
    "deberta-large-sst2": {"model_path": "Tomor0720/deberta-large-finetuned-sst2", "task": "zero-shot-classification"},
    "bart-large-sst2": {"model_path": "valhalla/bart-large-sst2", "task": "zero-shot-classification"},
    "electra-large-sst2": {"model_path": "howey/electra-large-sst2", "task": "zero-shot-classification"},
    # mrpc models
    "roberta-large-mrpc": {"model_path": "VitaliiVrublevskyi/roberta-large-finetuned-mrpc", "task": "text-classification"},
    "deberta-large-mrpc": {"model_path": "VitaliiVrublevskyi/deberta-large-finetuned-mrpc", "task": "text-classification"},
    "bart-large-mrpc": {"model_path": "Intel/bart-large-mrpc", "task": "text-classification"},
    "electra-large-mrpc": {"model_path": "titanbot/Electra-Large-MRPC", "task": "text-classification"},
    # cola models
    "roberta-large-cola": {"model_path": "iproskurina/tda-roberta-large-en-cola", "task": "text-classification"},
    "deberta-large-cola": {"model_path": "Tomor0720/deberta-large-finetuned-cola", "task": "text-classification"},
    "mobilebert-cola": {"model_path": "Alireza1044/mobilebert_cola", "task": "text-classification"},
    "electra-large-cola": {"model_path": "howey/electra-large-cola", "task": "text-classification"},
    "t5-large-cola": {"model_path": "thrunlab/t5-large_cola_sp0_05_ar0_0_one_router_from_pretrained_alpha1", "task": "text-classification"}
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
        choices=["mnli-m", "mnli-mm", "sst2", "qnli", "mrpc", "qqp", "cola", "rte"],
        help="Dataset to test on",
    )
    parser.add_argument(
        "--context_source",
        type=str,
        choices=["train", "validation"],
        help="Dataset to use for context",
    )
    parser.add_argument(
        "--test_source",
        type=str,
        choices=["validation", "test"],
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
    dataset = datasets.load_dataset("glue", dataset_name)
    label_list = dataset["train"].features["label"].names

    # the specific choice of demonstrations is determined by --seed
    if args.context_source == "train": 
        context_source = dataset[args.context_source].shuffle().select(range(args.num_examples))
    elif args.context_source == "validation":
        context_source = (
            dataset[args.context_source]
            if not args.dataset.startswith("mnli")
            else dataset[
                args.context_source + {"m": "_matched", "mm": "_mismatched"}[args.dataset[-1]]
            ]
        ).shuffle().select(range(args.num_examples))
    
    test_source = (
        dataset[args.test_source]
        if not args.dataset.startswith("mnli")
        else dataset[
            args.test_source + {"m": "_matched", "mm": "_mismatched"}[args.dataset[-1]]
        ]
    )
    # uncomment if you want to run code with just 100 examples (randomly sampled)
    # random_indices = random.sample(range(len(test_source)), 100)
    # test_source = test_source.select(random_indices)

    plugin_models_dict = {}
    for plugin_model in args.plugins:
        full_plugin_path = plugin_names[plugin_model]["model_path"]
        print(f"\nLoading plugin model: {plugin_model} ({full_plugin_path})")
        plugin_models_dict[plugin_model] = transformers.pipeline(plugin_names[plugin_model]["task"], model=full_plugin_path, device_map="cuda")

    plugin_data = {}          
    for i, example in enumerate(tqdm(context_source)):
        demonstration = f"{get_input_template(example, dataset_name)}\n"
        plugin_data[i] = {'demonstration': demonstration}
        plugin_input = get_plugin_template(example, dataset_name)
        for plugin_model_name, plugin_model in plugin_models_dict.items():
            if plugin_names[plugin_model_name]["task"] == "text-classification":
                plugin_model_result = plugin_model(plugin_input)[0]
                plugin_model_label = convert_label(plugin_model_result["label"], label_list)
                plugin_model_confidence = round(plugin_model_result["score"], 2)
            elif plugin_names[plugin_model_name]["task"] == "zero-shot-classification":
                plugin_model_result = plugin_model(plugin_input, label_list)
                plugin_model_label = plugin_model_result['labels'][0]
                plugin_model_confidence = round(plugin_model_result['scores'][0], 2)
            plugin_model_name_standard = standardize_model_name(plugin_model_name)
            plugin_string = f"{plugin_model_name_standard} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\n"
            plugin_data[i][plugin_model_name] = plugin_string
        true_label = label_list[example['label']]
        plugin_data[i]['label'] = 'Label: ' + true_label + '\n\n'
    

    file_name = f"{args.dataset}_context_{args.context_source}_{args.seed}"
    print(f'Saving results in data/plugin_data/{args.dataset}/{file_name}.json')
    with open(f'data/plugin_data/{args.dataset}/{file_name}.json', 'w') as f:
        json.dump(plugin_data, f)

    supericl_predictions = []
    supericl_ground_truth = []
    batch = []
    data = []

    plugin_data = {}
    for i, example in enumerate(tqdm(test_source)):
        valid_prompt = f"{get_input_template(example, dataset_name)}\n"
        plugin_data[i] = {'test_example': valid_prompt}
        plugin_input = get_plugin_template(example, dataset_name)
        for plugin_model_name, plugin_model in plugin_models_dict.items():
            if plugin_names[plugin_model_name]["task"] == "text-classification":
                plugin_model_result = plugin_model(plugin_input)[0]
                plugin_model_label = convert_label(plugin_model_result["label"], label_list)
                plugin_model_confidence = round(plugin_model_result["score"], 2)
            elif plugin_names[plugin_model_name]["task"] == "zero-shot-classification":
                plugin_model_result = plugin_model(plugin_input, label_list)
                plugin_model_label = plugin_model_result['labels'][0]
                plugin_model_confidence = round(plugin_model_result['scores'][0], 2)
            plugin_model_name_standard = standardize_model_name(plugin_model_name)
            plugin_string = f"{plugin_model_name_standard} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\n"
            plugin_data[i][plugin_model_name] = plugin_string
        plugin_data[i]['label'] = "Label: \n\n"

    file_name = f"{args.dataset}_{args.test_source}_examples"
    print(f'Saving results in data/plugin_data/{args.dataset}/{file_name}.json')
    with open(f'data/plugin_data/{args.dataset}/{file_name}.json', 'w') as f:
        json.dump(plugin_data, f)
