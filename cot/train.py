import argparse

from model_utils import load_model_dicts, load_model
from dataset import prep_data
from learner import train_model






def parse_option():
    parser = argparse.ArgumentParser(description="Testing various models")
    parser.add_argument('--model_id', default = 'bigscience/mt0-small', type=str, help='Model type')
    args = parser.parse_args()
    return args

def main(args):
    model_id = args.model_id

    m_dicts = load_model_dicts()
    model, tokenizer = load_model(model_id, m_dicts[model_id])

    prep_data(tokenizer)
    train_model(model, tokenizer)


if __name__ == "__main__":
    args = parse_option()
    print(args)
    main(args)