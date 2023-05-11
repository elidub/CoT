import argparse

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

from model_utils import load_model_dicts, load_model


def parse_option():
    parser = argparse.ArgumentParser(description="Plot renderings")
    parser.add_argument('--model_id', type=str, default='bigscience/mt0-small', help='model')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='max_new_tokens')
    args = parser.parse_args()
    return args


def main(args):

    model_id = args.model_id

    m_dicts = load_model_dicts()
    assert model_id in m_dicts.keys(), f"model_id {model_id} not in m_dicts.keys()"

    model, tokenizer = load_model(model_id = model_id, model_dict = m_dicts[model_id])

    inputs = tokenizer.encode("What is love?", return_tensors="pt").to('cuda')
    outputs = model.generate(inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(outputs[0]))

if __name__ == '__main__':
    args = parse_option()
    print(args)
    main(args)