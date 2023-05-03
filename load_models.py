from utils import load_model_dicts, load_model



if __name__ == '__main__':


    m_dicts = load_model_dicts()

    for m_id, m_dict in m_dicts.items():

        print(f'Loading model: {m_id}')

        model, tokenizer = load_model(m_id, m_dict)

        print(f'Vocab size: {tokenizer.vocab_size}.')
        print(f'Model parameters: {model.num_parameters}')