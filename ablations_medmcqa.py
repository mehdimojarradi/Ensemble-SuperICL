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

def generate_instruction(label_list, plugins):
    task = f"determining the medical subject that a given question belongs to ({', '.join(label_list)})"

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
        default="medmcqa",
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
    parser.add_argument(
        "--context_source",
        type=str,
        default="train",
        help="Dataset to use for context",
    )
    parser.add_argument(
        "--test_source",
        type=str,
        default="validation",
        help="Dataset to use for test input",
    )    
    args = parser.parse_args()
    full_llm_path = llm_names[args.llm]

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
    
    batch_size = 6 # modify as you wish to fit GPU memory

    print(f"\nLoading LLM: {args.llm} ({full_llm_path})")
    tokenizer, model = get_model(full_llm_path)

    context_json = f"data/plugin_data/{args.dataset}/{args.dataset}_context_train_{args.seed}_ftype_mnli.json"
    in_context_supericl_prompt = extract_super_icl_prompt(context_json, args.plugins, args.num_examples, no_conf=args.no_conf, no_context=args.no_context)

    supericl_predictions = []
    supericl_ground_truth = []
    batch = []
    data = []

    test_json = f"data/plugin_data/{args.dataset}/{args.dataset}_validation_examples_ftype_mnli.json"

    print(f"\nRunning {args.num_examples}-shot SuperICL on dataset: {args.dataset}, with LLM: {args.llm}, with plugin models: {args.plugins}")
    for i, q in enumerate(tqdm(test_questions)):
        
        valid_prompt = extract_test_prompt(test_json, args.plugins, i, no_conf=args.no_conf, no_test=args.no_test)

        instruction = generate_instruction(label_list, args.plugins)
        input_prompt = [
            {"role": "system", "content": instruction + in_context_supericl_prompt},
            {"role": "user", "content": valid_prompt},
            ]
        batch.append(input_prompt)
        if len(batch) == batch_size or i+1==len(test_questions):
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