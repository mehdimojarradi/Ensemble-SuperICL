import json
import re
import random
from collections import Counter
from sklearn.metrics import matthews_corrcoef, accuracy_score

def get_input_template(example, dataset_name):
    if dataset_name == "mnli":
        return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    elif dataset_name == "sst2":
        return f"Sentence: {example['sentence']}"
    elif dataset_name == "qnli":
        return f"Question: {example['question']}\nSentence: {example['sentence']}"
    elif dataset_name == "mrpc":
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}"
    elif dataset_name == "qqp":
        return f"Question 1: {example['question1']}\nQuestion 2: {example['question2']}"
    elif dataset_name == "cola":
        return f"Sentence: {example['sentence']}"
    elif dataset_name == "rte":
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}"
    elif dataset_name == "medmcqa":
        return f"Question: {example['question']}"


def get_plugin_template(example, dataset_name):
    if dataset_name == "mnli":
        return f"{example['premise']} <s> {example['hypothesis']}"
    elif dataset_name == "sst2":
        return f"{example['sentence']}"
    elif dataset_name == "qnli":
        return f"{example['question']} <s> {example['sentence']}"
    elif dataset_name == "mrpc":
        return f"{example['sentence1']} <s> {example['sentence2']}"
    elif dataset_name == "qqp":
        return f"{example['question1']} <s> {example['question2']}"
    elif dataset_name == "cola":
        return f"{example['sentence']}"
    elif dataset_name == "rte":
        return f"{example['sentence1']} <s> {example['sentence2']}"
    elif dataset_name == "medmcqa":
        return f"{example['question']}"
    
def get_subject_labels(questions):
    # for medmcqa dataset
    unique_subject_names = set()
    for q in questions:
        unique_subject_names.add(q['subject_name'])
    return list(unique_subject_names)
    
def standardize_model_name(plugin_model_name):
    if 'roberta-large' in plugin_model_name:
        return 'RoBERTa-Large'
    elif 'deberta-large' in plugin_model_name:
        return 'DeBERTa-Large'
    elif 'bart-large' in plugin_model_name:
        return 'BART-Large'
    elif 'flan-t5' in plugin_model_name:
        return 'flan-t5-base'
    elif 'electra-large' in plugin_model_name:
        return 'ELECTRA-Large'
    elif 't5-large' in plugin_model_name:
        return 'T5-Large'
    elif 'mobilebert' in plugin_model_name:
        return 'MobileBERT'
    else:
        raise ValueError(f"Invalid model name: {plugin_model_name}")
    
def convert_label(label, label_list):
    if label.startswith("LABEL_"):
        try:
            return label_list[int(label.split("_")[-1])]
        except IndexError:
            return "null"
    else:
        return label.lower()
    
def extract_super_icl_prompt(json_file, plugin_models, num_examples, no_conf = False, no_context = False):
    with open(json_file, 'r') as f:
        data = json.load(f)
        data = {int(k): v for k, v in data.items()}  # Convert keys to integers
    output = ""
    for i in range(num_examples):
        output += data[i]['demonstration']
        for plugin_model in plugin_models:
            if no_context == True:
                pass
            elif no_conf == True:
                model_output = data[i][plugin_model]
                model_output = re.sub(r' \(Confidence: [\d\.]+\)', '', model_output)
                output += model_output
            else:
                output += data[i][plugin_model]
        output += data[i]['label']
    return output

def extract_super_icl_prompt_no_conf(json_file, plugin_models, num_examples):
    with open(json_file, 'r') as f:
        data = json.load(f)
        data = {int(k): v for k, v in data.items()}  # Convert keys to integers
    output = ""
    for i in range(num_examples):
        output += data[i]['demonstration']
        for plugin_model in plugin_models:
            model_output = data[i][plugin_model]
            model_output = re.sub(r' \(Confidence: [\d\.]+\)', '', model_output)
            output += model_output
        output += data[i]['label']
    return output

def extract_test_prompt(json_file, plugin_models, idx, no_conf = False, no_test = False):
    with open(json_file, 'r') as f:
        data = json.load(f)
        data = {int(k): v for k, v in data.items()}  # Convert keys to integers
    output = ""
    output += data[idx]['test_example']
    for plugin_model in plugin_models:
        if no_test == True:
            pass
        elif no_conf == True:
            model_output = data[idx][plugin_model]
            model_output = re.sub(r' \(Confidence: [\d\.]+\)', '', model_output)
            output += model_output
        else:
            output += data[idx][plugin_model]
    output += data[idx]['label']
    return output

def extract_test_prompt_no_conf(json_file, plugin_models, idx):
    with open(json_file, 'r') as f:
        data = json.load(f)
        data = {int(k): v for k, v in data.items()}  # Convert keys to integers
    output = ""
    output += data[idx]['test_example']
    for plugin_model in plugin_models:
        model_output = data[idx][plugin_model]
        model_output = re.sub(r' \(Confidence: [\d\.]+\)', '', model_output)
        output += model_output
    output += data[idx]['label']
    return output

def extract_test_prompt_permute_conf(json_file, plugin_models, idx, conf_range=(0.5, 0.6)):
    with open(json_file, 'r') as f:
        data = json.load(f)
        data = {int(k): v for k, v in data.items()}  # Convert keys to integers
    output = ""
    output += data[idx]['test_example']
    for plugin_model in plugin_models:
        prediction = data[idx][plugin_model]
        new_confidence = random.uniform(*conf_range)
        prediction = re.sub(r'\(Confidence: .*\)', f'(Confidence: {new_confidence:.2f})', prediction)
        output += prediction
    output += data[idx]['label']
    return output

def extract_first_letters(plugins): # for creating the file at the end of a run
    return ''.join(plugin[0] for plugin in plugins)

def extract_finetune_type(plugins):
    for plugin in plugins:
        if 'mnli' in plugin:
            return 'mnli'
        elif 'sst2' in plugin:
            return 'sst2'
        elif 'mrpc' in plugin:
            return 'mrpc'
        elif 'cola' in plugin:
            return 'cola'
    return None

def extract_info(path):
    """
    Extracts specific substrings from a given path string. Useful for analysing results.

    Parameters:
    path (str): The input string, formatted like a path.

    Returns:
    str: A string containing the substrings found between 'supericl_' and '_llama', and between 'instruct_' and '_base', 
         separated by a space. If either substring is not found, returns None.

    Example:
    For the input "data/train_validation/supericl5/mnli-m/supericl_rdbfe_llama3-8b-instruct_24shot_mnli-m_base_format_instruction.json", 
    the function returns "rdbfe 24shot mnli-m".
    """
    pattern1 = r"supericl_(.*?)_llama"
    pattern2 = r"instruct_(.*?)_base"
    
    match1 = re.search(pattern1, path)
    match2 = re.search(pattern2, path)
    
    if match1 and match2:
        return match1.group(1) + ' ' + match2.group(1)
    else:
        return None