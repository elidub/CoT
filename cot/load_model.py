from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from utils import load_model_dicts, load_model
from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB")

if __name__ == '__main__':
    m_dicts = load_model_dicts()
    [print(m_id) for m_id, _ in m_dicts.items()];

    model, tokenizer = load_model(
            't5-small',
            {
                'model' : T5ForConditionalGeneration, 
                'tokenizer' : T5Tokenizer, 
                # 'model_kwargs' : {'device_map':'auto', 'load_in_8bit':True}, 
                # 'tokenizer_kwargs' : {}
            },
            hf_cache = '/nfs/scratch/atcs_cot/hf_cache/'
        )

    print_gpu_utilization()
