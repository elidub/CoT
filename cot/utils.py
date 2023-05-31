import pynvml
import json

def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB")

def load_model_dicts(models_yml = "CoT/models.yml"):
    with open("CoT/models.yml") as f:
        models = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in models.items():
        for key in ['model', 'tokenizer']:
            v[key] = transformer_dict[v[key]]

    return models

def load_model(model_id, model_dict, hf_cache = '/nfs/scratch/atcs_lcur1654/'):
    print(hf_cache)
    model = model_dict['model'].from_pretrained(model_id, **model_dict.get('model_kwargs', {}), cache_dir = hf_cache)
    tokenizer = model_dict['tokenizer'].from_pretrained(model_id, **model_dict.get('tokenizer_kwargs', {}), cache_dir = hf_cache)
    return model, tokenizer


def update_results(results = 'store/results.json', evals = None, args = None):
    # open store/results.json
    with open(results, 'r') as f:
        results_dict = json.load(f)

    args = vars(args)
    args['accuracy'] = evals['eval_accuracy']
    args['loss']     = evals['eval_loss']

    results_dict.update({args['save_name'] : args})

    # save store/results.json
    with open(results, 'w') as f:
        json.dump(results_dict, f)

