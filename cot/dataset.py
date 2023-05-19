import os
import torch
import random
import json
import pickle
import nltk
from pathlib import Path
from datasets import load_from_disk
from datasets import get_dataset_config_names, load_dataset
from bigbench_utils import load_bigbench_dataset

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


class CoTDataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, split="train"):
        self.config = config
        self.split = split
        self.tokenizer = tokenizer

        # If necessary, download, tokenize dataset and save to disk
        preprocessed_path = Path(self.config.preprocessed_dir) / self.preprocessed_filename()
        if self.config.rebuild_cache or not preprocessed_path.exists():
            # TODO: Look at notebook
            # https://colab.research.google.com/drive/1MKdLdF7oqrSQCeavAcsEnPdI85kD0LzU?usp=sharing#scrollTo=UqoBR0ujfEkl

            try:
                # local variant that contains truthful_qa
                dt = load_bigbench_dataset(self.config.bigbench_task_name, 'data/bigbench')[split]
            except NotImplementedError as e:
                # try the HF version if we don't have it locally, also this source only contains 50k samples per dataset max
                dt = load_dataset('tasksource/bigbench', self.config.bigbench_task_name, split=self.split, cache_dir=self.config.hf_cache_dir)


            tokenized_dataset = {}
            print(next(iter(dt)))
            
            # Tokenize
            tokenized_dataset["inputs"] = [self.tokenizer(sample["inputs"]) for sample in dt]
            tokenized_dataset["targets"] = [self.tokenizer(sample["targets"]) for sample in dt]
            # tokenized_dataset["inputs_untokenized"] = [sample["inputs"] for sample in dt]
            # tokenized_dataset["labels_untokenized"] = [sample["targets"] for sample in dt]
            
            # Save to disk
            os.makedirs(self.config.preprocessed_dir, exist_ok=True)
            with open(preprocessed_path, "wb") as file:
                pickle.dump(tokenized_dataset, file)
                    
        else:
            # Load dataset from disk
            tokenized_dataset = {}
            with open(preprocessed_path, "rb") as file:
                tokenized_dataset = pickle.load(file)


        self.data = tokenized_dataset["inputs"]
        self.labels = tokenized_dataset["targets"]

        # self.data = torch.Tensor(tokenized_dataset["inputs"])
        # self.labels = torch.Tensor(tokenized_dataset["targets"]['input_ids'])


        # print("Types of data and labels")
        # print(type(self.data))
        # print(type(self.labels))
        
        # self.untok_data = tokenized_dataset["inputs_untokenized"]
        # self.untok_labels = tokenized_dataset["labels_untokenized"]

        # Tokenize cot's
        handtuned_file_path = Path(self.config.bigbench_explanations_path) / "handtuned" / (self.config.bigbench_task_name + ".json")
        questions, answers, explanations = parse_handtuned(handtuned_file_path)
        # tokenized_explanations = [self.tokenizer(questions[i] + answers[i] + explanations[i]) for i in range(len(questions))]
        

        # Store cot's
        self.cots = [{
            "id": None,
            "tokenized_example" : self.tokenizer(questions[i] + answers[i] + explanations[i]),
            "example": questions[i] + answers[i] + explanations[i],
            "label": None,
        } for i in range(len(questions)) ]

    def preprocessed_filename(self):
        return self.config.bigbench_task_name + "_" + self.split + ".json"

    def __getitem__(self, idx):
        # Select CoT's (includes shuffling if dynamic)
        n = self.config.n_shot if self.config.n_shot <= 5 else 5

        if n > 0:
            if self.config.shuffle_cots:
                cot_idx = random.sample(range(5), n)
            else:
                cot_idx = list(range(n))

            concatenated_input_ids = []
            concatenated_attention_mask = []
            # concatenated_untokenized_string = []

            for i in cot_idx:
                concatenated_input_ids += self.cots[i]['tokenized_example']["input_ids"]
                concatenated_attention_mask += self.cots[i]['tokenized_example']["attention_mask"]
                # concatenated_untokenized_string += self.cots[i]['example']

            # Concatenate
            item = self.data[idx]

            #do we even need attention mask?
            concatenated_input_ids += item["input_ids"]
            concatenated_attention_mask += item["attention_mask"]
            # concatenated_untokenized_string += self.untok_data[idx]


            # Return x and y
            x = {
                'input_ids':  torch.Tensor(concatenated_input_ids).long(),
                'attention_mask':  torch.Tensor(concatenated_attention_mask).long(),
                'labels':  torch.Tensor(self.labels[idx]["input_ids"]).long().squeeze()
            }

        #0-shot
        else:

            x = {
                'input_ids': torch.Tensor(self.data[idx]["input_ids"]).long(),
                'attention_mask': torch.Tensor(self.data[idx]["attention_mask"]).long(),
                'labels':  torch.Tensor(self.labels[idx]["input_ids"]).long().squeeze()
            }

        return x

    def __len__(self):
        return len(self.labels)


# import argparse
# from model_utils import load_model_dicts, load_model

# args = argparse.Namespace()
# args.model_id = "bigscience/mt0-small"
# args.hf_cache_dir = "datadump/hf"
# args.preprocessed_dir = "datadump/preprocessed"
# args.bigbench_task_name = "odd_one_out"
# args.bigbench_explanations_path = "data/bigbench-explanations/"
# args.rebuild_cache = True
# args.shuffle_cots = False
# args.n_shot = 5


# model_id = args.model_id
# m_dicts = load_model_dicts()
# model, tokenizer = load_model(model_id, m_dicts[model_id])
# ds = CoTDataset(args, tokenizer, "train")
# item = next(iter(ds))
# print("Done!")