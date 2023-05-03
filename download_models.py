from utils import load_model_dicts, load_model



if __name__ == '__main__':


    m_dicts = load_model_dicts()

    for m_id, m_dict in m_dicts.items():

        print(f'Downloading model: {m_id}')

        # Don't download it with all the settings
        if 'model_kwargs' in m_dict:
            del m_dict['model_kwargs'] 

        model, tokenizer = load_model(m_id, m_dict)

        print(f'Vocab size: {tokenizer.vocab_size}.')
        print(f'Model parameters: {model.num_parameters}')