import torch
import datasets
import transformers
import random
import numpy as np
from tqdm.auto import tqdm
import argparse
import sys
from sklearn.metrics import accuracy_score
import json
from datetime import datetime

from utils import get_model, run_llama3
from helpers import *

def generate_instruction(label_list, plugins):
    task = f"determining the medical subject that a given question belongs to ({', '.join(label_list)})"

    if args.run_supericl:
        plugin_models = [standardize_model_name(model) for model in plugins]
        models_string = ', '.join(plugin_models)
        if len(plugin_models) > 1:
            instruction = f"You are tasked with {task}. {models_string} are language models fine-tuned on this task, and you may use their output as an aid to your judgement. Fill in your answer after 'Label: ' at the end of the prompt.\n\n"
        else:
            instruction = f"You are tasked with {task}. {models_string} is a language model fine-tuned on this task, and you may use its output as an aid to your judgement. Fill in your answer after 'Label: ' at the end of the prompt.\n\n"
    elif args.run_icl:
        instruction = f"You are tasked with {task}. Here are {args.num_examples} examples of the task with correct responses. Fill in your answer after 'Label: ' at the end of the prompt.\n\n"
    return instruction

def get_input_template2(q):
    return f"Question: {q['question']}"

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
        default= "medmcqa",
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
        "--run_supericl",
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
    dataset = datasets.load_dataset("openlifescienceai/medmcqa")
    context_questions = []
    test_questions = []

    for i, row in enumerate(dataset["train"]):
        if row['subject_name'] in ['Dental', 'Surgery']:
            context_questions.append(dataset["train"][i])
    
    ## creating a balanced dataset of context questions
    dental_questions = [q for q in context_questions if q['subject_name'] == 'Dental']
    surgery_questions = [q for q in context_questions if q['subject_name'] == 'Surgery']
    sample_dental_questions = random.sample(dental_questions, 350)
    sample_surgery_questions = random.sample(surgery_questions, 350)
    context_questions = sample_dental_questions + sample_surgery_questions
    random.shuffle(context_questions)
    context_questions = context_questions[:args.num_examples]

    for i, row in enumerate(dataset["validation"]):
        if row['subject_name'] in ['Dental', 'Surgery']:
            test_questions.append(dataset["validation"][i])

    ## creating a balanced dataset of test questions
    dental_questions = [q for q in test_questions if q['subject_name'] == 'Dental']
    surgery_questions = [q for q in test_questions if q['subject_name'] == 'Surgery']
    sample_dental_questions = random.sample(dental_questions, 350)
    sample_surgery_questions = random.sample(surgery_questions, 350)
    test_questions = sample_dental_questions + sample_surgery_questions

    label_list = get_subject_labels(test_questions)
    
    if args.run_plugins:
        plugin_models_dict = {}
        for plugin_model in args.plugins:
            full_plugin_path = plugin_names[plugin_model]["model_path"]
            print(f"\nLoading plugin model: {plugin_model} ({full_plugin_path})")
            plugin_models_dict[plugin_model] = transformers.pipeline(plugin_names[plugin_model]["task"], model=full_plugin_path, device_map="cuda")
        
        plugin_predictions = {plugin_model: [] for plugin_model in args.plugins}
        plugin_model_ground_truth = []
        print(f"\nTesting zero-shot inference on dataset: {args.dataset}, with small language models: {args.plugins}")
        for q in tqdm(test_questions):
            for plugin_model_name, plugin_model in plugin_models_dict.items():
                prompt = get_plugin_template(q, dataset_name)
                plugin_model_label = plugin_model(prompt, label_list)['labels'][0]
                plugin_predictions[plugin_model_name].append(plugin_model_label)
            plugin_model_ground_truth.append(q["subject_name"])

        for plugin_model_name in args.plugins:
            print(
                f"{plugin_model_name} Accuracy: {round(accuracy_score(plugin_predictions[plugin_model_name], plugin_model_ground_truth)*100, 3)}"
            )

    batch_size = args.batch_size
    if args.run_icl:
        print(f"\nLoading LLM: {args.llm} ({full_llm_path})")
        tokenizer, model = get_model(full_llm_path)
        in_context_prompts = [f"{get_input_template2(q)}\nLabel: {q['subject_name']}\n\n" for q in context_questions]
        in_context_prompt = "".join(in_context_prompts)

        icl_predictions = []
        icl_ground_truth = []
        batch = []
        data = []

        print(f"\nRunning {args.num_examples}-shot ICL on dataset: {args.dataset}, with LLM: {args.llm}")
        for i, q in enumerate(tqdm(test_questions)):
            instruction = generate_instruction(label_list, args.plugins)
            valid_prompt = f"{get_input_template2(q)}\nLabel: \n\n"
            input_prompt = [
                {"role": "system", "content": instruction + in_context_prompt},
                {"role": "user", "content": valid_prompt},
                ]
            batch.append(input_prompt)
            if len(batch) == batch_size or i+1==len(test_questions):
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
                    if last_line in label_list:
                        icl_prediction = last_line.lower()
                    elif last_line.startswith('Label: '):
                        icl_prediction = last_line[len('Label: '):].strip().lower()
                    else:
                        icl_prediction = lines[-3].lower()
                    icl_predictions.append(icl_prediction)
                    true_label = q['subject_name'].lower()
                    data.append({
                        "input_prompt": input_prompt,
                        "response": response,
                        "icl_prediction": icl_prediction,
                        "true_label": true_label
                    })
                batch = []
            icl_ground_truth.append(q['subject_name'].lower())

        ## adding useful information to the data file
        label_list = [label.lower() for label in label_list] # lowercasing for hallucinations
        data.append({'true_labels': icl_ground_truth})
        data.append({'icl_predictions': icl_predictions})
        hallucinations = [p for p in icl_predictions if p not in label_list]
        data.append({'hallucinations': hallucinations})
        hallucination_rate = round(len(hallucinations) / len(icl_predictions) * 100, 3)
        data.append({'hallucination_rate': hallucination_rate})
        accuracy = round(accuracy_score(icl_predictions, icl_ground_truth) * 100, 3)
        data.append({'accuracy_score': accuracy})

        datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data.append({'datetime': datetime})

        print(
            f"\n{args.num_examples}-shot ICL on dataset: {args.dataset}, with LLM: {args.llm}\nICL Accuracy: {accuracy}%\nHallucination rate: {hallucination_rate}%"
        )

        file_name = f"icl_{args.llm}_{args.num_examples}shot_{args.dataset}_base_format_instruction_{args.seed}"
        print(f'Saving results in data/medmcqa/train_test/icl/{file_name}.json')
        with open(f'data/medmcqa/train_test/icl/{file_name}.json', 'w') as f:
            json.dump(data, f)

    batch_size = args.batch_size

    if args.run_supericl:
        print(f"\nLoading LLM: {args.llm} ({full_llm_path})")
        tokenizer, model = get_model(full_llm_path)

        if dataset_name not in args.plugins:
            finetune_type = extract_finetune_type(args.plugins)
            context_json = f"data/plugin_data/medmcqa/adj_finetune/{args.dataset}_context_train_{args.seed}_ftype_{finetune_type}.json"
        else:
            context_json = f"data/plugin_data/{args.dataset}/{args.dataset}_context_train_{args.seed}.json"
        in_context_supericl_prompt = extract_super_icl_prompt(context_json, args.plugins, args.num_examples)

        supericl_predictions = []
        supericl_ground_truth = []
        batch = []
        data = []

        if dataset_name not in args.plugins:
            finetune_type = extract_finetune_type(args.plugins)
            test_json = f"data/plugin_data/medmcqa/adj_finetune/{args.dataset}_test_examples_ftype_{finetune_type}.json"
        else:
            test_json = f"data/plugin_data/{args.dataset}/{args.dataset}_test_examples.json"

        print(f"\nRunning {args.num_examples}-shot SuperICL on dataset: {args.dataset}, with LLM: {args.llm}, with plugin models: {args.plugins}")
        for i, q in enumerate(tqdm(test_questions)):
            instruction = generate_instruction(label_list, args.plugins)
            valid_prompt = extract_test_prompt(test_json, args.plugins, i)
            input_prompt = [
                {"role": "system", "content": instruction + in_context_supericl_prompt},
                {"role": "user", "content": valid_prompt},
                ]
            batch.append(input_prompt)
            if len(batch) == batch_size or i+1==len(test_questions):
                responses = run_llama3( # run_model is base, run_llama3 to use specified prompt format
                    tokenizer,
                    model,
                    prompt=batch,
                    temperature=1, # 1 is base and default, 0.01 is alt; Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic
                    max_new_tokens=10,
                    top_p=0.5, # 0.5 is base, 0.9 is alt, 1.0 is default; When decoding text, samples from the top p percentage of most likely tokens
                    # top_k=1, # default is 50; When decoding text, samples from the top k most likely tokens
                    repetition_penalty=1, #default
                    num_beams=1, #default
                    num_return_sequences=1 #default
                )
                for i, response in enumerate(responses["choices"]):
                    lines = response["text"].split('\n')
                    last_line = lines[-1]
                    if last_line in label_list:
                        supericl_prediction = last_line.lower()
                    elif last_line.startswith('Label: '):
                        supericl_prediction = last_line[len('Label: '):].strip().lower()
                    else:
                        supericl_prediction = lines[-3].lower()
                    supericl_predictions.append(supericl_prediction)
                    true_label = q['subject_name'].lower()
                    data.append({
                        "input_prompt": batch[i],
                        "response": response["text"], # check this
                        "supericl_prediction": supericl_prediction,
                        "true_label": true_label
                    })
                batch = []
            supericl_ground_truth.append(q['subject_name'].lower())            

        ## adding useful information to the data file
        label_list = [label.lower() for label in label_list] # lowercasing for hallucinations
        data.append({'true_labels': supericl_ground_truth})
        data.append({'supericl_predictions': supericl_predictions})
        hallucinations = [p for p in supericl_predictions if p not in label_list]
        data.append({'hallucinations': hallucinations})
        hallucination_rate = round(len(hallucinations) / len(supericl_predictions) * 100, 3)
        data.append({'hallucination_rate': hallucination_rate})
        accuracy = round(accuracy_score(supericl_predictions, supericl_ground_truth) * 100, 3)
        data.append({'accuracy_score': accuracy})
        
        datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data.append({'datetime': datetime})

        print(
            f"{args.num_examples}-shot SuperICL on dataset: {args.dataset}, with LLM: {args.llm}, with plugin models: {args.plugins}\nSuperICL Accuracy: {accuracy}%\nHallucination rate: {hallucination_rate}%"
        )

        plugins = extract_first_letters(args.plugins)
        directory = 'supericl' + str(len(args.plugins))
        file_name = f"supericl_{plugins}_{args.llm}_{args.num_examples}shot_{args.dataset}_base_format_instruction_{args.seed}"

        if dataset_name not in args.plugins:
            finetune_type = extract_finetune_type(args.plugins)
            file_name += f"_{finetune_type}_finetune"
            print(f'Saving results in data/zero_shot/medmcqa/{directory}/{file_name}.json')
            with open(f'data/zero_shot/medmcqa/{directory}/{file_name}.json', 'w') as f:
                json.dump(data, f)
        else:    
            print(f'Saving results in data/medmcqa/train_test/{directory}/{file_name}.json')
            with open(f'data/medmcqa/train_test/{directory}/{file_name}.json', 'w') as f:
                json.dump(data, f)