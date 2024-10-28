from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_model(llm):
    """
    Loads the tokenizer and model for a given language model.

    Args:
        llm (str): The name or path of the pre-trained language model.

    Returns:
        tuple: A tuple containing the tokenizer and the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        llm, trust_remote_code=True, use_fast=False, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        llm, trust_remote_code=True, device_map="cuda",
        torch_dtype=torch.bfloat16,
        load_in_8bit=True
    )

    return tokenizer, model

def run_llama3(tokenizer, model, *args, **kwargs):
    """
    Generates a response using the LLaMA-3 model based on the provided prompt.

    Args:
        tokenizer (AutoTokenizer): The tokenizer for the LLaMA-3 model.
        model (AutoModelForCausalLM): The LLaMA-3 model.
        *args: Additional positional arguments for the model's generate method.
        **kwargs: Additional keyword arguments for the model's generate method.
            - prompt (str): The input prompt for the model.

    Returns:
        dict: A dictionary containing the generated text response.
    
    how llama3 should be called: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
    """
    prompt = tokenizer.apply_chat_template(
        kwargs["prompt"],
        add_generation_prompt=True,
        return_dict=True,  # returns other tokenizer outputs like attention mask
        return_tensors="pt",
        padding="longest"
    )

    attn_mask = prompt["attention_mask"].to(model.device)
    input_ids = prompt["input_ids"].to(model.device)
        
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    del kwargs["prompt"]
    gen = model.generate(
        input_ids,
        attention_mask=attn_mask,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        do_sample=False,  # set as False for deterministic (reproducible) results
        **kwargs
    )
    
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
    response = {"choices": [{"text": txt} for txt in texts]}

    return response