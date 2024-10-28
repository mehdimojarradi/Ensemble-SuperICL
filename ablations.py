import torch
import datasets
import random
import numpy as np
from tqdm.auto import tqdm
import argparse
from sklearn.metrics import matthews_corrcoef, accuracy_score
import json
from datetime import datetime

from utils import get_model, run_llama3
from helpers import *

def generate_instruction(dataset, plugins):
    if dataset == "mrpc":
        task = "determining whether two sentences are semantically equivalent (equivalent or not_equivalent)"
    elif dataset == "sst2":
        task = "predicting the sentiment of a given sentence (positive or negative)"
    elif dataset == "mnli-m":
        task = "determining the relationship between a pair of sentences as entailment (the hypothesis is a true conclusion from the premise), contradiction (the hypothesis contradicts the premise), or neutral (the hypothesis neither necessarily follows from nor contradicts the premise)"
    elif dataset == "cola":
        task = "determining whether the grammar of a given sentence is correct (acceptable or unacceptable)"

    plugin_models = [standardize_model_name(model) for model in plugins]
    models_string = ', '.join(plugin_models)
    if args.no_test == True:
        instruction = f"You are tasked with {task}. Fill in your answer after 'Label: ' at the end of the prompt.\n\n"
    else:
        if len(plugin_models) > 1:
            instruction = f"You are tasked with {task}. {models_string} are language models fine-tuned on this task, and you may use their output as an aid to your judgement. Fill in your answer after 'Label: ' at the end of the prompt.\n\n"
        else:
            instruction = f"You are tasked with {task}. {models_string} is a language model fine-tuned on this task, and you may use its output as an aid to your judgement. Fill in your answer after 'Label: ' at the end of the prompt.\n\n"
    return instruction

if __name__ == "__main__":
    llm_names = {
    "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    }
    plugin_names = {
    # mnli models
    "roberta-large-mnli": "FacebookAI/roberta-large-mnli",
    "deberta-large-mnli": "microsoft/deberta-large-mnli",
    "bart-large-mnli": "facebook/bart-large-mnli",
    "flan-t5-mnli": "sjrhuschlee/flan-t5-base-mnli",
    "electra-large-mnli": "howey/electra-large-mnli",
    # sst2 models
    "roberta-large-sst2": "philschmid/roberta-large-sst2",
    "deberta-large-sst2": "Tomor0720/deberta-large-finetuned-sst2",
    "bart-large-sst2": "valhalla/bart-large-sst2",
    "electra-large-sst2": "howey/electra-large-sst2",
    # mrpc models
    "roberta-large-mrpc": "VitaliiVrublevskyi/roberta-large-finetuned-mrpc",
    "deberta-large-mrpc": "VitaliiVrublevskyi/deberta-large-finetuned-mrpc",
    "bart-large-mrpc": "Intel/bart-large-mrpc",
    "electra-large-mrpc": "titanbot/Electra-Large-MRPC",
    # cola models
    "roberta-large-cola": "iproskurina/tda-roberta-large-en-cola",
    "deberta-large-cola": "Tomor0720/deberta-large-finetuned-cola",
    "mobilebert-cola": "Alireza1044/mobilebert_cola",
    "electra-large-cola": "howey/electra-large-cola",
    "t5-large-cola": "thrunlab/t5-large_cola_sp0_05_ar0_0_one_router_from_pretrained_alpha1"
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        type=str,
        choices=list(llm_names.keys()),
        help="Path to LLM",
        )
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
        "--num_examples",
        type=int,
        default=32,
        help="Number of in-context examples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no_conf",
        action="store_true",
        default=False,
        help="remove confidence scores from prompt"
    )
    parser.add_argument(
        "--no_context",
        action="store_true",
        default=False,
        help="remove in-context examples from prompt"
    )
    parser.add_argument(
        "--no_test",
        action="store_true",
        default=False,
        help="remove SLM predictions from test input"
    )
    args = parser.parse_args()
    full_llm_path = llm_names[args.llm]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_name = args.dataset.split("-")[0]
    dataset = datasets.load_dataset("glue", dataset_name)
    label_list = dataset["train"].features["label"].names

    context_source = dataset["train"].shuffle().select(range(args.num_examples))
    test_source = (
        dataset["validation"]
        if not args.dataset.startswith("mnli")
        else dataset[
            "validation" + {"m": "_matched", "mm": "_mismatched"}[args.dataset[-1]]
        ]
    )
    # uncomment if you want to run code with just 100 examples (randomly sampled)
    # random_indices = random.sample(range(len(test_source)), 100)
    # test_source = test_source.select(random_indices)

    batch_size = 6 # modify as you wish to fit GPU memory

    print(f"\nLoading LLM: {args.llm} ({full_llm_path})")
    tokenizer, model = get_model(full_llm_path)

    context_json = f"data/plugin_data/{args.dataset}/{args.dataset}_context_train_{args.seed}.json"
    in_context_supericl_prompt = extract_super_icl_prompt(context_json, args.plugins, args.num_examples, no_conf=args.no_conf, no_context=args.no_context)

    supericl_predictions = []
    supericl_ground_truth = []
    batch = []
    data = []

    test_json = f"data/plugin_data/{args.dataset}/{args.dataset}_validation_examples.json"

    print(f"\nRunning {args.num_examples}-shot SuperICL on dataset: {args.dataset}, with LLM: {args.llm}, with plugin models: {args.plugins}")
    for i, example in enumerate(tqdm(test_source)):

        valid_prompt = extract_test_prompt(test_json, args.plugins, i, no_conf=args.no_conf, no_test=args.no_test)

        instruction = generate_instruction(args.dataset, args.plugins)
        input_prompt = [
            {"role": "system", "content": instruction
            + in_context_supericl_prompt
            },
            {"role": "user", "content": valid_prompt},
            ]
        batch.append(input_prompt)
        if len(batch) == batch_size or i+1==len(test_source):
            responses = run_llama3(
                tokenizer,
                model,
                prompt=batch,
                temperature=1, # adjusts randomness of outputs, greater than 1 is random and 0 is deterministic
                max_new_tokens=10,
                top_p=0.5, # when decoding text, samples from the top p percentage of most likely tokens
                # top_k=1, when decoding text, samples from the top k most likely tokens
                repetition_penalty=1, #default
                num_beams=1, #default
                num_return_sequences=1 #default
            )
            for i, response in enumerate(responses["choices"]):
                lines = response["text"].split('\n')
                last_line = lines[-1]
                if last_line.startswith('Label: '):
                    supericl_prediction = last_line[len('Label: '):].strip().lower()
                else:
                    supericl_prediction = last_line.lower()
                supericl_predictions.append(supericl_prediction)
                true_label = label_list[example["label"]]
                data.append({
                    "input_prompt": batch[i],
                    "response": response["text"],
                    "supericl_prediction": supericl_prediction,
                    "true_label": true_label
                })
            batch = []
        supericl_ground_truth.append(label_list[example["label"]])            

    ## adding useful information to the data file
    data.append({'true_labels': supericl_ground_truth})
    data.append({'supericl_predictions': supericl_predictions})
    hallucinations = [p for p in supericl_predictions if p not in label_list]
    data.append({'hallucinations': hallucinations})
    hallucination_rate = round(len(hallucinations) / len(supericl_predictions) * 100, 3)
    data.append({'hallucination_rate': hallucination_rate})

    if dataset_name == "cola":
        accuracy = round(matthews_corrcoef(supericl_predictions, supericl_ground_truth) * 100, 3)
    else:
        accuracy = round(accuracy_score(supericl_predictions, supericl_ground_truth) * 100, 3)
    data.append({'accuracy_score': accuracy})
    
    datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data.append({'datetime': datetime})

    if dataset_name == "cola":
        print(
            f"{args.num_examples}-shot SuperICL on dataset: {args.dataset}, with LLM: {args.llm}, with plugin models: {args.plugins}\nSuperICL Matthews Corr: {accuracy}%\nHallucination rate: {hallucination_rate}%"
        )
    else:
        print(
            f"{args.num_examples}-shot SuperICL on dataset: {args.dataset}, with LLM: {args.llm}, with plugin models: {args.plugins}\nSuperICL Accuracy: {accuracy}%\nHallucination rate: {hallucination_rate}%"
        )

    plugins = extract_first_letters(args.plugins)
    # directory = 'supericl' + str(len(args.plugins))
    file_name = f"supericl_{plugins}_{args.llm}_{args.num_examples}shot_{args.dataset}"
    if args.no_context:
        file_name += "_no_context"
    if args.no_conf:
        file_name += "_no_conf"
    if args.no_test:
        file_name += "_no_test"
    print(f'Saving results in data/ablations/{args.dataset}/{file_name}.json')
    with open(f'data/ablations/{args.dataset}/{file_name}.json', 'w') as f:
        json.dump(data, f)