import os
import torch
import random
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from datasets import get_dataset_config_names, load_dataset
from bigbench_utils import load_bigbench_dataset, filter_arithmetic_tasks
from sklearn.model_selection import train_test_split

# SETUP
# os.system('pip install git+https://github.com/google/BIG-bench.git')
# os.system('pip install datasets')

#parse a handtuned explanations json file
def parse_handtuned(file_path, bigbench_explanations_dataset):

    with open(file_path, 'r') as file:
        data = json.load(file)
        for key in ["hand_tuned_fewshot_explanations_0", "hand_tuned_fewshot_explanations_1", "hand_tuned_fewshot_explanations_2"]:
            if key in data:
                full_string = data[key]

    q_list = []
    a_list = []
    explanation_list = []

    # Identify how questions, explanations, and answers are introduced in this dataset
    if bigbench_explanations_dataset == "presuppositions_as_nli":
        sample_prefix = "Sentence 1: "
        answer_prefix = "The answer is: "
        explanation_prefix = "Explanation: "
    else:
        sample_prefix = "Q: "
        answer_prefix = "A: "
        explanation_prefix = "Explanation: "

    assert full_string.count(sample_prefix) == full_string.count(answer_prefix) == full_string.count(explanation_prefix) == 5, "Unexpected formatting of CoTs"

    substrings = full_string.split(sample_prefix)[1:]  # Split the string and discard the first element
    for substring in substrings:
        q, a_explanation = substring.split("\n" + answer_prefix, 1)  # Split each substring into question and remaining part
        a, explanation = a_explanation.split("\n" + explanation_prefix, 1)  # Split the remaining part into answer and explanation

        q_list.append(sample_prefix + q.strip() + "\n")  # Add "Q: " prefix and strip any leading/trailing whitespaces
        a_list.append(answer_prefix + a.strip() + "\n")  # Add "A: " prefix and strip any leading/trailing whitespaces
        explanation_list.append(explanation_prefix + explanation.strip() + "\n")  # Add "Explanation: " prefix and strip any leading/trailing whitespaces

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

            # Load dataset for fine-tuning
            if self.config.dataset_is_bigbench:
                # Train and validation overlap, so we load train set and extract required split
                try:
                    # local variant that contains truthful_qa
                    dt = load_bigbench_dataset(self.config.dataset_name, 'data/bigbench')["train"]
                except NotImplementedError as e:
                    # try the HF version if we don't have it locally, also this source only contains 50k samples per dataset max
                    dt = load_dataset('tasksource/bigbench', self.config.dataset_name, split="train", cache_dir=self.config.hf_cache_dir)

                # If specified, arithmetic datasets can be filtered to one specific operation
                if self.config.dataset_name == "arithmetic" and self.config.arithmetic_task_name:
                    dt = filter_arithmetic_tasks(dt, self.config.arithmetic_task_name)

                train, val = train_test_split(dt, test_size=0.3, random_state=self.config.seed)
                dt = val if self.split=="validation" else train

            
            #qea
            # Remove 'A:' from the end of the question and append 'Explanation: '
            if self.config.qae:
                if dt['inputs'][0][-2:] == "A:":
                    for sample in range(len(dt['inputs'])):
                        dt['inputs'][sample] = dt['inputs'][sample][:-2] + "Explanation: "

                elif dt['inputs'][0][-3:] == "A: ":
                    for sample in range(len(dt['inputs'])):
                        dt['inputs'][sample] = dt['inputs'][sample][:-3] + "Explanation: "
                
                elif not dt['inputs'][0].endswith("Explanation:") or not dt['inputs'][0].endswith("Explanation: "):
                    for sample in range(len(dt['inputs'])):
                        dt['inputs'][sample] = dt['inputs'][sample] + "Explanation: "
                
        

            tokenized_dataset = {}
            
            if type(dt) == list:
                # Tokenize
                tokenized_dataset["inputs"] = [self.tokenizer(sample["inputs"]) for sample in dt]
                tokenized_dataset["targets"] = [self.tokenizer(sample["targets"]) for sample in dt]
                if self.config.debug:
                    tokenized_dataset["inputs_untokenized"] = [sample["inputs"] for sample in dt]
                    tokenized_dataset["labels_untokenized"] = [sample["targets"] for sample in dt]
            elif type(dt) == dict:
                # Tokenize
                tokenized_dataset["inputs"] = [self.tokenizer(sample) for sample in dt["inputs"]]
                tokenized_dataset["targets"] = [self.tokenizer(sample) for sample in dt["targets"]]
                if self.config.debug:
                    tokenized_dataset["inputs_untokenized"] = [sample for sample in dt["inputs"]]
                    tokenized_dataset["labels_untokenized"] = [sample for sample in dt["targets"]]
            else:
                raise NotImplementedError("Unknown type of dataset")

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

        # print(tokenized_dataset['inputs'][0])
        # print(tokenized_dataset['targets'][0])


        
        if self.config.debug:
            self.untok_data = tokenized_dataset["inputs_untokenized"]
            self.untok_labels = tokenized_dataset["labels_untokenized"]
            # print("AAAAAAAAAAAAAAAAAAAAAAa")
            # print(tokenized_dataset['inputs_untokenized'][0])
            # print(tokenized_dataset['labels_untokenized'][0])

        # Tokenize cot's
        handtuned_file_path = Path(self.config.bigbench_explanations_path) / self.config.bigbench_explanations_type / (self.config.bigbench_explanations_dataset + ".json")
        questions, answers, explanations = parse_handtuned(handtuned_file_path, self.config.bigbench_explanations_dataset)
        # tokenized_explanations = [self.tokenizer(questions[i] + answers[i] + explanations[i]) for i in range(len(questions))]



        #TODO! CHECK IF CONFIG.QAE IS TRUE WHEN APPENDING STEP BY STEP

        # In zero-shot CoT setting we have the option to append "Let's think this step by step"
        if self.config.step_by_step and self.config.n_shot == 0:
            self.step_by_step = self.tokenizer("Let's think this step by step.")

            # Append the tokenized step_by_step string BEFORE A:
            for item in self.data:
                item["input_ids"] = item["input_ids"][:-3] + self.step_by_step["input_ids"] + item["input_ids"][-3:-1]
                item["attention_mask"] = item["attention_mask"][:-3] + self.step_by_step["attention_mask"] + item["attention_mask"][-3:-1]
                # item["token_type_ids"] = item["token_type_ids"][:-3] + self.step_by_step["token_type_ids"] + item["token_type_ids"][-3:-1]

        
        # Store cot's
        if self.config.qae:
            self.cots = [{
                "id": None,
                "tokenized_example" : self.tokenizer(questions[i] + answers[i] + explanations[i]),
                "example": questions[i] + answers[i] + explanations[i],
                "label": None,
            } for i in range(len(questions)) ]
            
        else:
            self.cots = [{
                "id": None,
                "tokenized_example" : self.tokenizer(questions[i] + explanations[i] + answers[i]),
                "example": questions[i] + explanations[i] + answers[i],
                "label": None,
            } for i in range(len(questions)) ]


        # Print some information about the dataset
        if self.config.debug:
            first_sample = self[0]
            # print(f"First preprocessed sample: {first_sample=}")

        if self.config.debug:
            # print(f"Computing longest required context for {self.split}...")
            longest_sample = max(self, key=lambda x: len(x['input_ids']) + len(x['labels']))
            max_tokens = len(longest_sample['input_ids']) + len(longest_sample['labels'])
            # print(f"Longest sample ({max_tokens} tokens): {longest_sample}...")

    def preprocessed_filename(self):
        return self.config.dataset_name + "_" + self.split + ("_debug" if self.config.debug else "") + ".json"

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

            for i in cot_idx:
                concatenated_input_ids += self.cots[i]['tokenized_example']["input_ids"]
                concatenated_attention_mask += self.cots[i]['tokenized_example']["attention_mask"]

            # Concatenate
            item = self.data[idx]

            #do we even need attention mask?
            concatenated_input_ids += item["input_ids"]
            concatenated_attention_mask += item["attention_mask"]

            if self.split == 'validation' and self.config.dataset_name == 'squad':
                x = {
                    'input_ids':  torch.Tensor(concatenated_input_ids).long(),
                    'attention_mask':  torch.Tensor(concatenated_attention_mask).long(),
                    'labels':  torch.Tensor(self.labels[idx]["input_ids"][0]).long().squeeze()
                }
            else:
                # Return x and y
                x = {
                    'input_ids':  torch.Tensor(concatenated_input_ids).long(),
                    'attention_mask':  torch.Tensor(concatenated_attention_mask).long(),
                    'labels':  torch.Tensor(self.labels[idx]["input_ids"]).long().squeeze()
                }

            # When debugging, additionally store untokenized data
            if self.config.debug:
                concatenated_untokenized_string = ""
                for i in cot_idx:
                    concatenated_untokenized_string += self.cots[i]['example']
                concatenated_untokenized_string += self.untok_data[idx]
                x["untokenized_inputs"] = concatenated_untokenized_string
                x["untokenized_labels"] = self.untok_labels[idx]

        #0-shot
        else:
            x = {
                'input_ids':  torch.Tensor(item["input_ids"]).long(),
                'attention_mask':  torch.Tensor(item["attention_mask"]).long(),
                'labels':  torch.Tensor(self.labels[idx]["input_ids"]).long().squeeze()
            }

            # When debugging, additionally store untokenized data
            if self.config.debug:
                x["untokenized_inputs"] = self.untok_data[idx]
                x["untokenized_labels"] = self.untok_labels[idx]

        return x

    def __len__(self):
        return len(self.labels)


# import argparse
# from model_utils import load_model_dicts, load_model

# args = argparse.Namespace()
# args.model_id = "bigscience/mt0-small"
# args.hf_cache_dir = "datadump/hf"
# args.debug = True
# args.model_max_length = None

# args.preprocessed_dir = "datadump/preprocessed"

# # Example 1 for fine-tuning on bigbench:
# args.dataset_name = "presuppositions_as_nli"
# args.dataset_is_bigbench = True
# args.bigbench_explanations_dataset = "presuppositions_as_nli"
# args.model_max_length = 1015  # Length for 5-shot

# Example 2 for fine-tuning on bigbench:
# args.dataset_name = "truthful_qa"
# args.dataset_is_bigbench = True
# args.bigbench_explanations_dataset = "truthful_qa"

# # Example 3 for fine-tuning on bigbench:
# args.dataset_name = "arithmetic"
# args.arithmetic_task_name = "3_digit_division"
# args.dataset_is_bigbench = True
# args.bigbench_explanations_dataset = "arithmetic_3_digit_division"

# # Example for fine-tuning on squad:
# args.dataset_name = "squad"
# args.dataset_is_bigbench = False
# args.bigbench_explanations_dataset = "truthful_qa"


# args.bigbench_explanations_type = "handtuned"
# args.bigbench_explanations_path = "data/bigbench-explanations/"
# args.rebuild_cache = True
# args.shuffle_cots = False
# args.n_shot = 5

# args.lr = 1e-3
# args.max_epochs = 100
# args.batch_size = 8
# args.seed = 666

# args.lora_r = 8
# args.lora_alpha = 32
# args.lora_dropout = 0.05
# args.lora_bias = "none"

# model_id = args.model_id
# m_dicts = load_model_dicts()
# model, tokenizer = load_model(model_id, m_dicts[model_id], model_max_length=args.model_max_length)
# ds = CoTDataset(args, tokenizer, "validation")
# item = next(iter(ds))
# print("Done!")