import argparse

from model_utils import load_model_dicts, load_model
from learner import train_model
from dataset import CoTDataset






def parse_option():
    parser = argparse.ArgumentParser(description="Testing various models")

    # General args
    parser.add_argument('--model_id', default = 'bigscience/mt0-small', type=str, help='Model type')
    parser.add_argument('--hf_cache_dir', default = '/project/gpuuva021/shared/cot/hf_cache', type=str, help='Directory for HuggingFace cache')
    parser.add_argument('--debug', action='store_true', help='Use smaller datasets and store more info for debugging')

    # Dataset args
    parser.add_argument('--preprocessed_dir', default = '/project/gpuuva021/shared/cot/data/preprocessed', type=str, help='Directory for storing the preprocessed datasets')
    parser.add_argument('--dataset_name', default = 'squad', type=str, help='The name of the dataset on huggingface on which models are fine-tuned and evaluated.')
    parser.add_argument('--dataset_is_bigbench', action='store_true', help='Use this flag if the dataset specified by dataset_name is part of the bigbench collection.')
    parser.add_argument('--bigbench_explanations_dataset', default = 'truthful_qa', type=str, help='The name of the bigbench task from which the CoTs are taken.')
    parser.add_argument('--bigbench_explanations_type', default = 'handtuned', type=str, help='Type of explanations from Lampinen et al. One of "handtuned", "selected", "untuned".')
    parser.add_argument('--bigbench_explanations_path', default = 'data/bigbench-explanations/', type=str, help='Path to the bigbench explanations from Lampinen et al.')
    parser.add_argument('--n_shot', default = 3, type=int, help='How many examples to show in-context')
    parser.add_argument('--rebuild_cache', action='store_true', help='Whether to rebuild the cached preprocessed datasets')
    parser.add_argument('--shuffle_cots', default = False, type=bool, help='Whether to randomly select the available CoTs and their order. If False, the first n_shot CoTs are chosen.')

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

def main(args):
    model_id = args.model_id

    m_dicts = load_model_dicts()
    model, tokenizer = load_model(model_id, m_dicts[model_id], hf_cache=args.hf_cache_dir)

    tokenized_datasets = {}
    for split in ["train", "validation"]:
        tokenized_datasets[split] = CoTDataset(args, tokenizer, split)

    train_model(model, tokenizer, tokenized_datasets, args)


if __name__ == "__main__":
    debug = False
    if debug:
        args = argparse.Namespace()
        args.model_id = "bigscience/mt0-small"
        args.hf_cache_dir = "datadump/hf"
        args.debug = True

        args.preprocessed_dir = "datadump/preprocessed"

        # Example for fine-tuning on squad:
        # args.dataset_name = "squad"
        # args.dataset_is_bigbench = False

        # Example for fine-tuning on bigbench:
        args.dataset_name = "disambiguation_qa"
        args.dataset_is_bigbench = True

        args.bigbench_explanations_dataset = "truthful_qa"
        args.bigbench_explanations_type = "handtuned"
        args.bigbench_explanations_path = "data/bigbench-explanations/"
        args.rebuild_cache = False
        args.shuffle_cots = False
        args.n_shot = 5

        args.lr = 1e-3
        args.max_epochs = 100
        args.batch_size = 8
        args.seed = 666

        args.lora_r = 8
        args.lora_alpha = 32
        args.lora_dropout = 0.05
        args.lora_bias = "none"
    else:
        args = parse_option()
    print(args)
    main(args)