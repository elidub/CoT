import os
import torch
from datasets import load_from_disk
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoTokenizer


# SETUP
# os.system('pip install git+https://github.com/google/BIG-bench.git')
# os.system('pip install datasets')

config = {
    "raw_data_path": None,
    "cot_path": None,
    "preprocessed_dir": None,
    "bigbench_task_name": None,
    "bigbench_explanations_directory": None,
    "tokenizer_name": None,

    "rebuild_cache": False,

    "n_shot": 0,

}

class CoTDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config

        self.tokenizer = None

        # If necessary, download, tokenize dataset and save to disk
        preprocessed_path = self.config["preprocessed_dir"] + "/" + self.preprocessed_filename()
        if config["rebuild_cache"] or os.path.exists(preprocessed_path):
            # TODO: Look at notebook
            # https://colab.research.google.com/drive/1MKdLdF7oqrSQCeavAcsEnPdI85kD0LzU?usp=sharing#scrollTo=UqoBR0ujfEkl

            dt = load_dataset('bigbench', self.config["bigbench_task_name"])
            tokenized_dataset = {}
            for split in dt:
                tokenized_dataset[split] = {}
                if split=="default" and "train" in dt:
                    continue
                
                # Tokenize
                # (Some tasks might require task-dependent formatting)
                tokenized_dataset[split]["inputs"] = [self.tokenizer(sample["inputs"]) for sample in dt[split]]
                tokenized_dataset[split]["targets"] = [self.tokenizer(sample["targets"]) for sample in dt[split]]
                
                # Save to disk
                # TODO


        # Load dataset from disk

        # Tokenize cot's

        # Store cot's
        self.cots = [{
            "id": None,
            "tokenized_explanation" : None,
            "explanation": None,
            "label": None,
        }]

        pass

    def preprocessed_filename(self):
        return ""

    def __getitem__(self, idx):
        # Select CoT's (includes shuffling if dynamic)

        # Concatenate 

        # Return x and y
        pass

    def __len__(self):
        return len(self.data.labels)


class CoTDataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset):
        self.dataset = dataset

    def collate(self, batch):
        pass


# Testing
test_ds = CoTDataset(config)