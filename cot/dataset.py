import os
import torch
import random
import json
import pickle

from datasets import load_from_disk
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoTokenizer


# SETUP
# os.system('pip install git+https://github.com/google/BIG-bench.git')
# os.system('pip install datasets')

#parse a handtuned explanations json file
def parse_handtuned(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
        for key in ["hand_tuned_fewshot_explanations_0", "hand_tuned_fewshot_explanations_1", "hand_tuned_fewshot_explanations_2"]:
            if key in data:
                string = data[key]

    q_list = []
    a_list = []
    explanation_list = []

    substrings = string.split("Q: ")[1:]  # Split the string and discard the first element

    for substring in substrings:
        q, a_explanation = substring.split("\nA: ", 1)  # Split each substring into question and remaining part
        a, explanation = a_explanation.split("\nExplanation: ", 1)  # Split the remaining part into answer and explanation

        q_list.append("Q: " + q.strip() + "\n")  # Add "Q: " prefix and strip any leading/trailing whitespaces
        a_list.append("A: " + a.strip() + "\n")  # Add "A: " prefix and strip any leading/trailing whitespaces
        explanation_list.append("Explanation: " + explanation.strip() + "\n")  # Add "Explanation: " prefix and strip any leading/trailing whitespaces

    return q_list, a_list, explanation_list

# Example usage
# path = "../../bigben/handtuned/arithmetic_3_digit_division.json"     
# q_list, a_list, explanation_list = parse_handtuned(path)

config = {
    "raw_data_path": None,
    "cot_path": None,
    "preprocessed_dir": None,
    "bigbench_task_name": None,
    "bigbench_explanations_path": None,
    "tokenizer_name": None,
    "rebuild_cache": False,
    "n_shot": 0,
    "split": None,
    "shuffle" : False,


}

class CoTDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config

        #this will probably be given as a hyperparameter
        self.tokenizer = None

        # If necessary, download, tokenize dataset and save to disk
        preprocessed_path = self.config["preprocessed_dir"] + "/" + self.preprocessed_filename()
        if config["rebuild_cache"] or os.path.exists(preprocessed_path):
            # TODO: Look at notebook
            # https://colab.research.google.com/drive/1MKdLdF7oqrSQCeavAcsEnPdI85kD0LzU?usp=sharing#scrollTo=UqoBR0ujfEkl

            dt = load_dataset('bigbench', self.config["bigbench_task_name"], split=config.split)
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
                split_file_path = os.path.join(preprocessed_path, f"{split}.pkl")
                with open(split_file_path, "wb") as file:
                    pickle.dump(tokenized_dataset[split], file)
                    
        else:
            # Load dataset from disk
            tokenized_dataset = {}
            split_file_path = os.path.join(preprocessed_path, f"{config.split}.pkl")
            with open(split_file_path, "rb") as file:
                tokenized_dataset[config.split] = pickle.load(file)

        self.data = tokenized_dataset[config.split]["inputs"]
        self.labels = tokenized_dataset[config.split]["targets"]

        # Tokenize cot's
        questions, answers, explanations = parse_handtuned(config.bigbench_explanations_path)
        # tokenized_explanations = [self.tokenizer(questions[i] + answers[i] + explanations[i]) for i in range(len(questions))]
        

        # Store cot's
        self.cots = [{
            "id": None,
            "tokenized_explanation" : self.tokenizer(questions[i] + answers[i] + explanations[i]),
            "explanation": questions[i] + answers[i] + explanations[i],
            "label": None,
        } for i in range(len(questions)) ]

        pass

    def preprocessed_filename(self):
        return ""

    def __getitem__(self, idx):
        # Select CoT's (includes shuffling if dynamic)
        n = self.config['n_shot'] if self.config['n_shot'] <= 5 else 5

        if n > 0:
            if self.config.shuffle:
                cot_idx = random.sample(range(5), n)
            else:
                cot_idx = list(range(n))

            concatenated_input_ids = []
            concatenated_attention_mask = []

            for i in cot_idx:
                concatenated_input_ids += self.cots[i]['tokenized_explanation']["input_ids"]
                concatenated_attention_mask += self.cots[i]['tokenized_explanation']["attention_mask"]

            # Concatenate
            item = self.data[idx]

            #do we even need attention mask?
            concatenated_input_ids = item["input_ids"] + concatenated_input_ids
            concatenated_attention_mask = item["attention_mask"] + concatenated_attention_mask


            # Return x and y
            x = {
                'input_ids': concatenated_input_ids,
                'attention_mask': concatenated_attention_mask,
                'labels': self.labels[idx]
            }
            y = self.labels[idx]

        #0-shot
        else:
            x = self.data[idx]
            y = self.labels[idx]
            x = {
                'input_ids': self.data[idx]["input_ids"],
                'attention_mask': self.data[idx]["attention_mask"],
                'labels': self.labels[idx]
            }

        return (x,y)

    def __len__(self):
        return len(self.data.labels)


class CoTDataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset):
        self.dataset = dataset

    def collate(self, batch):
        pass


# Testing
test_ds = CoTDataset(config)