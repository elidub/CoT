from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import yaml
import torch

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