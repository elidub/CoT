from utils import load_model_dicts, load_model

def run():
    m_dicts = load_model_dicts()
    [print(m_id) for m_id, _ in m_dicts.items()];
    
    m_id = 'google/flan-t5-small'
    model, tokenizer = load_model(m_id, m_dicts[m_id], hf_cache = '/nfs/scratch/atcs_cot/hf_cache_dev/')

    input_text = "translate English to German: How old are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))


if __name__ == '__main__':
    run()