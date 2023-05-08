from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

from utils import load_model_dicts, load_model

m_dicts = load_model_dicts()
[print(m_id) for m_id, _ in m_dicts.items()]
    
model, tokenizer = load_model(
        'google/flan-t5-small',
        {
            'model' : T5ForConditionalGeneration, 
            'tokenizer' : T5Tokenizer, 
            # 'model_kwargs' : {'device_map':'auto', 'load_in_8bit':True}, 
            # 'tokenizer_kwargs' : {}
        },
        hf_cache = '/nfs/scratch/atcs_cot2/hf_cache/'
    )

print(model)