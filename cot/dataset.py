import os
import torch
import json

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
        # TODO

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