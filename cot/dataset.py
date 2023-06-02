import os
import torch
import random
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from datasets import get_dataset_config_names, load_dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(1, sys.path[0] + '/../')
from cot.bigbench_utils import load_bigbench_dataset, filter_arithmetic_tasks

# SETUP
# os.system('pip install git+https://github.com/google/BIG-bench.git')
# os.system('pip install datasets')


# Example usage
# path = "../../bigben/handtuned/arithmetic_3_digit_division.json"     
# q_list, a_list, explanation_list = parse_handtuned(path)


class CoTDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split

        # If necessary, download, tokenize dataset and save to disk
        preprocessed_path = Path(self.args.preprocessed_dir) / self.preprocessed_filename()
        if self.args.rebuild_cache or not preprocessed_path.exists():

            # Load dataset for fine-tuning
            if self.args.dataset_is_bigbench:
                # Train and validation overlap, so we load train set and extract required split
                try:
                    # local variant that contains truthful_qa
                    dt = load_bigbench_dataset(self.args.dataset_name, 'data/bigbench')["train"]
                except NotImplementedError as e:
                    # try the HF version if we don't have it locally, also this source only contains 50k samples per dataset max
                    dt = load_dataset('tasksource/bigbench', self.args.dataset_name, split="train", cache_dir=self.args.hf_cache_dir)

                # If specified, arithmetic datasets can be filtered to one specific operation
                if self.args.dataset_name == "arithmetic" and self.args.arithmetic_task_name:
                    dt = filter_arithmetic_tasks(dt, self.args.arithmetic_task_name)
                else:
                    # Transform into subset even when it's not necessary, just to keep the code below consistent
                    # Is it dirty? Yes. Does it work? I hope so.
                    dt = Subset(dt, range(len(dt)))

                # # Uncomment for small dataset for debugging
                # print('Making a subset of the dataset')
                # dt = Subset(dt, range(100))

                train, val = train_test_split(dt, test_size=0.3, random_state=self.args.seed)
                dt = val if self.split=="validation" else train

            
            self.sample_prefix = "Q:"
            self.answer_prefix = "A:"
            self.explanation_prefix = "Explanation:"

            # Identify how questions, explanations, and answers are introduced in this dataset
            if self.args.bigbench_explanations_dataset == "presuppositions_as_nli":
                self.sample_prefix = "Sentence 1:"
                for i in range(len(dt)):
                    dt[i]['inputs'] = dt[i]['inputs'].replace("A:", "1:")
                    dt[i]['inputs'] = dt[i]['inputs'].replace("B:", "2:")
                    dt[i]['inputs'] = dt[i]['inputs'].replace("\nThe answer is:", "A:")

            if args.step_by_step:
                assert self.args.n_shot == 0, "Step by step is not compatible with n_shot > 0"
            
                
            if self.args.qae:
                # no reformatting needed for QAE
                pass 
            else:
                if self.args.n_shot > 0 and not self.args.no_explanation:
                    for i in range(len(dt)):
                        answer_prefix_index = dt[i]['inputs'].rfind(self.answer_prefix)
                        dt[i]['inputs'] = dt[i]['inputs'][:answer_prefix_index] # sample_without_answer_prefix
                        dt[i]['inputs'] += self.explanation_prefix

                if self.args.step_by_step:
                    for i in range(len(dt)):
                        answer_prefix_index = dt[i]['inputs'].rfind(self.answer_prefix)
                        dt[i]['inputs'] = dt[i]['inputs'][:answer_prefix_index] + "Let's think step by step.\n" + dt[i]['inputs'][answer_prefix_index:]


            #In nli, reposition the task explanation string
            if self.args.bigbench_explanations_dataset == "presuppositions_as_nli":
                task_string = 'Q: This is a natural language inference task. There are two sentences in English. The answer is "entailment" if the first sentence entails the second, "contradiction" if the second sentence contradicts the first, and "neutral" if neither is of those two cases holds.\n\n\n'
                for i in range(len(dt)):
                    dt[i]['inputs'] = dt[i]['inputs'].replace(task_string, "")

                    # #if we want to put the string at the top
                    # index = dt[i]['inputs'].find(task_string)
                    # if index != -1:
                    #     dt[i]['inputs'] = task_string + dt[i]['inputs'][:index] + dt[i]['inputs'][index + len(task_string):] 


            tokenized_dataset = {}
            
            if type(dt) == list:
                # Tokenize
                tokenized_dataset["inputs"] = [self.tokenizer(sample["inputs"]) for sample in dt]
                tokenized_dataset["targets"] = [self.tokenizer(sample["targets"]) for sample in dt]
                tokenized_dataset["inputs_untokenized"] = [sample["inputs"] for sample in dt]
                tokenized_dataset["labels_untokenized"] = [sample["targets"] for sample in dt]
            elif type(dt) == dict:
                # Tokenize
                tokenized_dataset["inputs"] = [self.tokenizer(sample) for sample in dt["inputs"]]
                tokenized_dataset["targets"] = [self.tokenizer(sample) for sample in dt["targets"]]
                tokenized_dataset["inputs_untokenized"] = [sample for sample in dt["inputs"]]
                tokenized_dataset["labels_untokenized"] = [sample for sample in dt["targets"]]
            else:
                raise NotImplementedError("Unknown type of dataset")
            

            # Save to disk
            os.makedirs(self.args.preprocessed_dir, exist_ok=True)
            with open(preprocessed_path, "wb") as file:
                pickle.dump(tokenized_dataset, file)
                    
        else:
            # Load dataset from disk
            tokenized_dataset = {}
            with open(preprocessed_path, "rb") as file:
                tokenized_dataset = pickle.load(file)

        # self.data = tokenized_dataset["inputs"] # Commenting out because we should use the untokenized data
        self.labels = tokenized_dataset["targets"]
        self.untok_data = tokenized_dataset["inputs_untokenized"]
        self.untok_labels = tokenized_dataset["labels_untokenized"]


        # Tokenize cot's
        handtuned_file_path = Path(self.args.bigbench_explanations_path) / self.args.bigbench_explanations_type / (self.args.bigbench_explanations_dataset + ".json")
        questions, answers, explanations = self.parse_handtuned(handtuned_file_path)

        def example(i):
            assert not (self.args.qae and self.args.no_explanation)
            if self.args.qae:
                return questions[i] + answers[i] + explanations[i]
            if self.args.no_explanation:
                return questions[i] + answers[i]
            else:
                return questions[i] + explanations[i] + answers[i]

        # Store cot's
        self.cots = [{
            "id": None,
            "tokenized_example" : self.tokenizer(example(i)),
            "example": example(i),
            "label": None,
        } for i in range(len(questions)) ]
            
    def preprocessed_filename(self):
        return self.args.dataset_name + "_" + self.split + ("_debug" if self.args.debug else "") + ".json"

    #parse a handtuned explanations json file
    def parse_handtuned(self, file_path):

        with open(file_path, 'r') as file:
            data = json.load(file)
            for key in ["hand_tuned_fewshot_explanations_0", "hand_tuned_fewshot_explanations_1", "hand_tuned_fewshot_explanations_2"]:
                if key in data:
                    full_string = data[key]

        if self.args.bigbench_explanations_dataset == "presuppositions_as_nli":
            full_string = full_string.replace("A:", "1:")
            full_string = full_string.replace("B:", "2:")
            full_string = full_string.replace("The answer is:", "A:")

        q_list = []
        a_list = []
        explanation_list = []

        assert full_string.count(self.sample_prefix) == full_string.count(self.answer_prefix) == full_string.count(self.explanation_prefix) == 5, "Unexpected formatting of CoTs"

        substrings = full_string.split(self.sample_prefix)[1:]  # Split the string and discard the first element
        for substring in substrings:
            q, a_explanation = substring.split("\n" + self.answer_prefix, 1)  # Split each substring into question and remaining part
            a, explanation = a_explanation.split("\n" + self.explanation_prefix, 1)  # Split the remaining part into answer and explanation

            q_list.append(self.sample_prefix + q.strip() + "\n")  # Add "Q: " prefix and strip any leading/trailing whitespaces
            a_list.append(self.answer_prefix + a.strip() + "\n")  # Add "A: " prefix and strip any leading/trailing whitespaces

            explanation_list.append(self.explanation_prefix + explanation.strip() + "\n")  # Add "Explanation: " prefix and strip any leading/trailing whitespaces

        return q_list, a_list, explanation_list

    def __getitem__(self, idx):
        # Select CoT's (includes shuffling if dynamic)
        n = self.args.n_shot if self.args.n_shot <= 5 else 5

        cot_idx = random.sample(range(5), n) if self.args.shuffle_cots else list(range(n))

        full_untokenized_sample = ""
        
        # Prepending CoTs, also works if n_shot == 0
        for i in cot_idx:
            full_untokenized_sample += self.cots[i]['example']

        full_untokenized_sample += self.untok_data[idx]
        full_tokenized_sample = self.tokenizer(full_untokenized_sample)["input_ids"]
        full_tokenized_label = self.tokenizer(self.untok_labels[idx][0])["input_ids"]

        if len(full_tokenized_label) < 3:
            full_tokenized_label.extend([-100] * (3 - len(full_tokenized_label)))

        x = {
                'input_ids':          torch.tensor(full_tokenized_sample).long(),
                'labels':             torch.tensor(full_tokenized_label).long(),
                'untokenized_sample': full_untokenized_sample,
                'labels_untokenized': self.untok_labels[idx]
            }
        return x
            
    def __len__(self):
        return len(self.labels)

