import torch
import datasets
import transformers
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

    if args.run_esupericl:
        plugin_models = [standardize_model_name(model) for model in plugins]
        models_string = ', '.join(plugin_models)
        if len(plugin_models) > 1:
            instruction = f"You are tasked with {task}. {models_string} are language models fine-tuned on this task, and you may use their output as an aid to your judgement. Fill in your answer after 'Label: ' at the end of the prompt.\n\n"
        else:
            instruction = f"You are tasked with {task}. {models_string} is a language model fine-tuned on this task, and you may use its output as an aid to your judgement. Fill in your answer after 'Label: ' at the end of the prompt.\n\n"
    elif args.run_icl:
        instruction = f"You are tasked with {task}. Here are {args.num_examples} examples of the task with correct responses. Fill in your answer after 'Label: ' at the end of the prompt.\n\n"
    return instruction

if __name__ == "__main__":
    llm_names = {
    "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    }
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
        choices=["mnli-m", "sst2", "qnli", "mrpc", "qqp", "cola", "rte"],
        help="Dataset to test on",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=32,
        help="Number of in-context examples"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size of examples for inference time"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--run_icl",
        action="store_true",
        default=False,
        help="Run ICL baseline"
    )
    parser.add_argument(
        "--run_plugins",
        action="store_true",
        default=False,
        help="Run plugin model baseline",
    )
    parser.add_argument(
        "--run_esupericl",
        action="store_true",
        default=False,
        help="Run SuperICL"
    )
    args = parser.parse_args()
    if args.llm:
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
    # uncomment if you want to test-run the code with just 100 examples (randomly sampled)
    # random_indices = random.sample(range(len(test_source)), 100)
    # test_source = test_source.select(random_indices)

    if args.run_plugins:
        plugin_models_dict = {}
        for plugin_model in args.plugins:
            full_plugin_path = plugin_names[plugin_model]["model_path"]
            print(f"\nLoading plugin model: {plugin_model} ({full_plugin_path})")
            plugin_models_dict[plugin_model] = transformers.pipeline(plugin_names[plugin_model]["task"], model=full_plugin_path, device_map="cuda")

        plugin_predictions = {plugin_model: [] for plugin_model in args.plugins}
        plugin_model_ground_truth = []
        print(f"\nTesting zero-shot inference on dataset: {args.dataset}, with small language models: {args.plugins}")
        for example in tqdm(test_source):
            for plugin_model_name, plugin_model in plugin_models_dict.items():
                if plugin_names[plugin_model_name]["task"] == "text-classification":
                    plugin_model_label = convert_label(
                        plugin_model(get_plugin_template(example, dataset_name))[0]["label"],
                        label_list,
                    )
                elif plugin_names[plugin_model_name]["task"] == "zero-shot-classification":
                    prompt = get_plugin_template(example, dataset_name)
                    plugin_model_label = plugin_model(prompt, label_list)['labels'][0]
                plugin_predictions[plugin_model_name].append(plugin_model_label)
            plugin_model_ground_truth.append(label_list[example["label"]])

        for plugin_model_name in args.plugins:
            if dataset_name == "cola":
                print(
                    f"{plugin_model_name} Matthews Corr: {round(matthews_corrcoef(plugin_predictions[plugin_model_name], plugin_model_ground_truth)*100, 3)}"
                )
            else:
                print(
                    f"{plugin_model_name} Accuracy: {round(accuracy_score(plugin_predictions[plugin_model_name], plugin_model_ground_truth)*100, 3)}"
                )

    batch_size = args.batch_size
    if args.run_icl:
        print(f"\nLoading LLM: {args.llm} ({full_llm_path})")
        tokenizer, model = get_model(full_llm_path)
        in_context_prompts = [f"{get_input_template(example, dataset_name)}\nLabel: {label_list[example['label']]}\n\n" for example in context_source]
        in_context_prompt = "".join(in_context_prompts)

        icl_predictions = []
        icl_ground_truth = []
        batch = []
        data = []

        print(f"\nRunning {args.num_examples}-shot ICL on dataset: {args.dataset}, with LLM: {args.llm}")
        for i, example in enumerate(tqdm(test_source)):
            instruction = generate_instruction(args.dataset, args.plugins)
            valid_prompt = f"{get_input_template(example, dataset_name)}\nLabel: \n\n"
            input_prompt = [
                {"role": "system", "content": instruction + in_context_prompt},
                {"role": "user", "content": valid_prompt},
                ]
            batch.append(input_prompt)
            if len(batch) == batch_size or i+1==len(test_source):
                responses = run_llama3( # run_model is base, run_llama3 to use specified prompt format
                    tokenizer,
                    model,
                    prompt=batch,
                    temperature=1, # 1 is base and default, 0.01 is alt; Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic
                    max_new_tokens=10,
                    top_p=0.5, # 0.5 is base, 0.9 is alt, 1.0 is default; When decoding text, samples from the top p percentage of most likely tokens
                    # top_k=1, # default is 50; When decoding text, samples from the top k most likely tokens
                    repetition_penalty=1,
                    num_beams=1,
                    num_return_sequences=1
                )
                for i, response in enumerate(responses["choices"]):
                    lines = response["text"].split('\n')
                    last_line = lines[-1]
                    if last_line.startswith('Label: '):
                        icl_prediction = last_line[len('Label: '):].strip().lower()
                    else:
                        icl_prediction = last_line.lower()
                    icl_predictions.append(icl_prediction)
                    true_label = label_list[example["label"]]
                    data.append({
                        "input_prompt": input_prompt,
                        "response": response,
                        "icl_prediction": icl_prediction,
                        "true_label": true_label
                    })
                batch = []
            icl_ground_truth.append(label_list[example["label"]])

        ## adding useful information to the data file
        data.append({'true_labels': icl_ground_truth})
        data.append({'icl_predictions': icl_predictions})
        hallucinations = [p for p in icl_predictions if p not in label_list]
        data.append({'hallucinations': hallucinations})
        hallucination_rate = round(len(hallucinations) / len(icl_predictions) * 100, 3)
        data.append({'hallucination_rate': hallucination_rate})

        if dataset_name == "cola":
            accuracy = round(matthews_corrcoef(icl_predictions, icl_ground_truth) * 100, 3)
        else:
            accuracy = round(accuracy_score(icl_predictions, icl_ground_truth) * 100, 3)
        data.append({'accuracy_score': accuracy})

        datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data.append({'datetime': datetime})

        if dataset_name == "cola":
            print(
                f"\n{args.num_examples}-shot ICL on dataset: {args.dataset}, with LLM: {args.llm}\nICL Matthews Corr: {accuracy}%\nHallucination rate: {hallucination_rate}%"
            )
        else:
            print(
                f"\n{args.num_examples}-shot ICL on dataset: {args.dataset}, with LLM: {args.llm}\nICL Accuracy: {accuracy}%\nHallucination rate: {hallucination_rate}%"
            )

        file_name = f"icl_{args.llm}_{args.num_examples}shot_{args.dataset}_{args.seed}"
        print(f'Saving results in data/icl/{args.dataset}/{file_name}.json')
        with open(f'data/icl/{args.dataset}/{file_name}.json', 'w') as f:
            json.dump(data, f)

    batch_size = args.batch_size

    if args.run_esupericl:
        print(f"\nLoading LLM: {args.llm} ({full_llm_path})")
        tokenizer, model = get_model(full_llm_path)

        context_json = f"data/plugin_data/{args.dataset}/{args.dataset}_context_{args.context_source}_{args.seed}.json"
        in_context_supericl_prompt = extract_super_icl_prompt(context_json, args.plugins, args.num_examples)

        supericl_predictions = []
        supericl_ground_truth = []
        batch = []
        data = []

        test_json = f"data/plugin_data/{args.dataset}/{args.dataset}_{args.test_source}_examples.json"

        print(f"\nRunning {args.num_examples}-shot SuperICL on dataset: {args.dataset}, with LLM: {args.llm}, with plugin models: {args.plugins}")
        for i, example in enumerate(tqdm(test_source)):
            valid_prompt = extract_test_prompt(test_json, args.plugins, i)

            instruction = generate_instruction(args.dataset, args.plugins)
            input_prompt = [
                {"role": "system", "content": instruction + in_context_supericl_prompt},
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
                    # top_k=1, # when decoding text, samples from the top k most likely tokens
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
        directory = 'supericl' + str(len(args.plugins))
        file_name = f"supericl_{plugins}_{args.llm}_{args.num_examples}shot_{args.dataset}{args.seed}"

        print(f'Saving results in data/{directory}/{args.dataset}/{file_name}.json')
        with open(f'data/{directory}/{args.dataset}/{file_name}.json', 'w') as f:
            json.dump(data, f)