from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import yaml
import torch

from datasets import load_from_disk
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainingArguments, DataCollator, DataCollatorWithPadding, DataCollatorForLanguageModeling
import wandb
import evaluate
import numpy as np
import torch
from torch import nn
from undecorated import undecorated
from types import MethodType

transformer_dict = {
    'AutoModelForCausalLM' : AutoModelForCausalLM,
    'AutoModelForSeq2SeqLM' : AutoModelForSeq2SeqLM,
    'AutoTokenizer' : AutoTokenizer,
    'T5ForConditionalGeneration' : T5ForConditionalGeneration,
    'T5Tokenizer' : T5Tokenizer
}

def load_model_dicts(models_yml = 'models.yml'):
    with open("models.yml") as f:
        models = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in models.items():
        for key in ['model', 'tokenizer']:
            v[key] = transformer_dict[v[key]]

    return models

def load_model(
        model_id,
        model_dict, 
        hf_cache = '/project/gpuuva021/shared/cot/hf_cache',
        model_kwargs = {
            'device_map' : 'auto',
            'load_in_8bit' : True,
            'torch_dtype' : torch.float16 # Overriding torch_dtype=None with `torch_dtype=torch.float16` due to requirements of `bitsandbytes` to enable model loading in mixed int8. Either pass torch_dtype=torch.float16 or don't pass this argument at all to remove this warning.
        },
        model_max_length = None,
    ):

    print(hf_cache)
    model = model_dict['model'].from_pretrained(
        model_id, 
        **model_kwargs,
    )
    tokenizer = model_dict['tokenizer'].from_pretrained(model_id, cache_dir = hf_cache, model_max_length=model_max_length)
    return model, tokenizer

def prep_lora_model(
        model,
        train,
        args,
        target_modules = {"mt0": ["q", "v"], "bloom": ["query_key_value"], "t5": ["q", "v"], "mt5": ["q", "v"]}
):
    # adapted from https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/peft-flan-t5-int8-summarization.ipynb
    # see TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING ub peft/utils/other.py
    if "mt0" in args.model_id:
        target_modules = target_modules["mt0"]
    elif "bloom-" in args.model_id:
        target_modules = target_modules["bloom"]
    elif "mt5" in args.model_id:
        target_modules = target_modules["mt5"]
    elif "t5" in args.model_id:
        target_modules = target_modules["t5"]
    else:
        raise NotImplementedError(f"no target modules specified for {args.model_id}")
    
    if train is False and args.wandb_run is None:
        print('No finetuning or evaluation run specified, this is an ablation run, not using LoRA!')
        return model
    
    # Define LoRA Config 
    if train is True:
        print('Setting up LoRA!')
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=TaskType.CAUSAL_LM
        )
        # prepare int-8 model for training
        model = prepare_model_for_int8_training(model)

    else:
        print('Loading trained LoRA!')
        save_dir = os.path.join("trained_models", args.wandb_run)
        lora_config = LoraConfig.from_pretrained(save_dir)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # create method for generating text without teacher forcing but with gradients
    # this is achieved by removing the "no grad" decorator from the base_model of the peft

    # unwrapped = model.base_model.generate.__wrapped__
    # model.generate = MethodType(unwrapped, model.base_model)

    return model