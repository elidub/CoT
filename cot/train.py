import argparse

from model_utils import load_model_dicts, load_model
from learner import train_model
from dataset import CoTDataset






def parse_option():
    parser = argparse.ArgumentParser(description="Testing various models")
    parser.add_argument('--model_id', default = 'bigscience/mt0-small', type=str, help='Model type')
    parser.add_argument('--hf_cache_dir', default = '/project/gpuuva021/shared/cot/hf_cache', type=str, help='Directory for HuggingFace cache')

    parser.add_argument('--preprocessed_dir', default = '/project/gpuuva021/shared/cot/data/preprocessed', type=str, help='Directory for storing the preprocessed datasets')
    parser.add_argument('--bigbench_task_name', default = 'truthful_qa', type=str, help='The name of the bigbench task on which to train and evaluate')
    parser.add_argument('--bigbench_explanations_path', default = 'data/bigbench-explanations/', type=str, help='Path to the bigbench explanations from Lampinen et al.')
    parser.add_argument('--n_shot', default = False, type=int, help='How many examples to show in-context')
    parser.add_argument('--rebuild_cache', default = False, type=bool, help='Whether to rebuild the cached preprocessed datasets')
    parser.add_argument('--shuffle_cots', default = False, type=bool, help='Whether to randomly select the available CoTs and their order. If False, the first n_shot CoTs are chosen.')
    args = parser.parse_args()
    return args

def main(args):
    model_id = args.model_id

    m_dicts = load_model_dicts()
    model, tokenizer = load_model(model_id, m_dicts[model_id])

    tokenized_dataset = {}
    for split in ["train", "validation"]:
        tokenized_dataset[split] = CoTDataset(args, tokenizer, split)

    train_model(model, tokenizer, args)


if __name__ == "__main__":
    debug = True
    if debug:
        args = argparse.Namespace()
        args.model_id = "bigscience/mt0-small"
        args.hf_cache_dir = "/project/gpuuva021/shared/cot/hf_cache"
        args.preprocessed_dir = "/project/gpuuva021/shared/cot/data/preprocessed"
        args.bigbench_task_name = "truthful_qa"
        args.bigbench_explanations_path = "data/bigbench-explanations/"
        args.rebuild_cache = False
        args.shuffle_cots = False
        args.n_shot = 5
    else:
        args = parse_option()
    print(args)
    main(args)