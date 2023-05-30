import argparse
import os, sys

# src_path = os.path.join(sys.path[0], '../')
# sys.path.insert(1, src_path)

sys.path.insert(1, sys.path[0] + '/../')


from cot.model_utils import load_model_dicts, load_model
from cot.learner import run_model
from cot.dataset import CoTDataset






def parse_option():
    parser = argparse.ArgumentParser(description="Testing various models")

    # General args
    parser.add_argument('--train', action = 'store_true', help='If True, train the model. If False, evaluate the model.')
    parser.add_argument('--wandb_run', default = None, help='WandB run ID, if None, a new run is created')
    parser.add_argument('--model_id', default = 'bigscience/mt0-small', type=str, help='Model type')
    parser.add_argument('--hf_cache_dir', default = '/project/gpuuva021/shared/cot/hf_cache', type=str, help='Directory for HuggingFace cache')
    parser.add_argument('--debug', action='store_true', help='Use smaller datasets and store more info for debugging')
    parser.add_argument('--model_max_length', default = 512, type=int, help='Longest context the model/tokenizer is expected to support')

    # Dataset args
    parser.add_argument('--preprocessed_dir', default = '/project/gpuuva021/shared/cot/data/preprocessed', type=str, help='Directory for storing the preprocessed datasets')
    parser.add_argument('--dataset_name', default = 'squad', type=str, help='The name of the dataset on huggingface on which models are fine-tuned and evaluated.')
    parser.add_argument('--dataset_is_bigbench', action='store_true', help='Use this flag if the dataset specified by dataset_name is part of the bigbench collection.')
    parser.add_argument('--arithmetic_task_name', required=False, default = None, type=str, help='If the dataset_name is arithmetic, then this arg can be used to filter to a specific arithmetic task.')
    parser.add_argument('--bigbench_explanations_dataset', default = 'odd_one_out', type=str, help='The name of the bigbench task from which the CoTs are taken.')
    parser.add_argument('--bigbench_explanations_type', default = 'handtuned', type=str, help='Type of explanations from Lampinen et al. One of "handtuned", "selected", "untuned".')
    parser.add_argument('--bigbench_explanations_path', default = 'data/bigbench-explanations/', type=str, help='Path to the bigbench explanations from Lampinen et al.')
    parser.add_argument('--n_shot', default = 3, type=int, help='How many examples to show in-context')
    parser.add_argument('--rebuild_cache', default = True, type=bool, help='Whether to rebuild the cached preprocessed datasets')
    parser.add_argument('--shuffle_cots', default = False, type=bool, help='Whether to randomly select the available CoTs and their order. If False, the first n_shot CoTs are chosen.')
    parser.add_argument('--step_by_step', action='store_true', help='Whether to append Lets think this step by step')
    parser.add_argument('--no_explanation', action='store_true', help='No explanations in the prefix')
    parser.add_argument('--qae', action='store_true', help='If True, the answer comes before the explanation')

    # Training args
    parser.add_argument('--lr', default = 1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_epochs', default = 10, type=int, help='Maximum number of epochs to train')
    parser.add_argument('--batch_size', default = 32, type=int, help='Batch size')
    parser.add_argument('--seed', default=666, type=int, help="The seed for reproducibility")

    # Training args (lora)
    parser.add_argument('--lora_r', default = 8, type=int, help='Rank of LoRa')
    parser.add_argument('--lora_alpha', default = 32, type=int, help='Alpha used for LoRa')
    parser.add_argument('--lora_dropout', default = 0.05, type=float, help='Alpha used for LoRa')
    parser.add_argument('--lora_bias', default = "none", type=str, help='Bias type for LoRa Can be "none", "all" or "lora_only"')

    args = parser.parse_args()
    return args

def debug_parse_option(notebook = False):
    parser = argparse.ArgumentParser(description="Testing various models")

    # General args
    parser.add_argument('--train', action = 'store_true', help='If True, train the model. If False, evaluate the model.')
    parser.add_argument('--wandb_run', default = None, help='WandB run ID, if None, a new run is created')
    parser.add_argument('--model_id', default = 'bigscience/bloom-1b1', type=str, help='Model type')
    parser.add_argument('--hf_cache_dir', default = 'datadump/hf', type=str, help='Directory for HuggingFace cache')
    parser.add_argument('--debug', default = True, help='Use smaller datasets and store more info for debugging')
    parser.add_argument('--model_max_length', default = 512, type=int, help='Longest context the model/tokenizer is expected to support')

    # Dataset args
    parser.add_argument('--preprocessed_dir', default = 'datadump/preprocessed', type=str, help='Directory for storing the preprocessed datasets')
    parser.add_argument('--dataset_name', default = 'arithmetic', type=str, help='The name of the dataset on huggingface on which models are fine-tuned and evaluated.')
    parser.add_argument('--dataset_is_bigbench', default = True, help='Use this flag if the dataset specified by dataset_name is part of the bigbench collection.')
    parser.add_argument('--arithmetic_task_name', default = '3_digit_division', type=str, help='If the dataset_name is arithmetic, then this arg can be used to filter to a specific arithmetic task.')
    parser.add_argument('--bigbench_explanations_dataset', default = 'arithmetic_3_digit_division', type=str, help='The name of the bigbench task from which the CoTs are taken.')
    parser.add_argument('--bigbench_explanations_type', default = 'handtuned', type=str, help='Type of explanations from Lampinen et al. One of "handtuned", "selected", "untuned".')
    parser.add_argument('--bigbench_explanations_path', default = 'data/bigbench-explanations/', type=str, help='Path to the bigbench explanations from Lampinen et al.')
    parser.add_argument('--n_shot', default = 0, type=int, help='How many examples to show in-context')
    parser.add_argument('--rebuild_cache', default = True, type=bool, help='Whether to rebuild the cached preprocessed datasets')
    parser.add_argument('--shuffle_cots', action='store_true', help='Whether to randomly select the available CoTs and their order. If False, the first n_shot CoTs are chosen.')
    parser.add_argument('--step_by_step', action='store_true', help='Append Lets think this step by step')
    parser.add_argument('--no_explanation', action='store_true', help='No explanations in the prefix')
    parser.add_argument('--qae', action='store_true', help='If True, the answer comes before the explanation')

    # Training args
    parser.add_argument('--lr', default = 1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_epochs', default = 1, type=int, help='Maximum number of epochs to train')
    parser.add_argument('--batch_size', default = 8, type=int, help='Batch size')
    parser.add_argument('--seed', default=666, type=int, help="The seed for reproducibility")

    # Training args (lora)
    parser.add_argument('--lora_r', default = 8, type=int, help='Rank of LoRa')
    parser.add_argument('--lora_alpha', default = 32, type=int, help='Alpha used for LoRa')
    parser.add_argument('--lora_dropout', default = 0.05, type=float, help='Alpha used for LoRa')
    parser.add_argument('--lora_bias', default = "none", type=str, help='Bias type for LoRa Can be "none", "all" or "lora_only"')

    args = parser.parse_args() if not notebook else parser.parse_args(args=[])
    return args

def main(args):
    m_dicts = load_model_dicts()
    model, tokenizer = load_model(args.model_id, m_dicts[args.model_id], hf_cache=args.hf_cache_dir, model_max_length=args.model_max_length)

    tokenized_datasets = {}
    for split in ["train", "validation"]:
        tokenized_datasets[split] = CoTDataset(args, tokenizer, split)

    run_model(model, tokenizer, tokenized_datasets, args)


if __name__ == "__main__":
    debug = True
    args = debug_parse_option () if debug else parse_option()
    print(args)
    main(args)


