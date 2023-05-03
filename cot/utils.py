from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import yaml

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

def load_model(model_id, model_dict, hf_cache = '/nfs/scratch/atcs_cot/hf_cache/'):
    print(hf_cache)
    model = model_dict['model'].from_pretrained(model_id, **model_dict.get('model_kwargs', {}), cache_dir = hf_cache)
    tokenizer = model_dict['tokenizer'].from_pretrained(model_id, **model_dict.get('tokenizer_kwargs', {}), cache_dir = hf_cache)
    return model, tokenizer