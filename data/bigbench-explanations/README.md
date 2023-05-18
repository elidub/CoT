# A few explanations for some BIG-Bench tasks

BIG-Bench (https://github.com/google/BIG-bench) is a collaboratively constructed benchmark dataset containing several hundred tasks. For our paper on explanations for language models (https://arxiv.org/abs/2204.02329) we selected a subset of 40 of these tasks, and annotated 15 randomly-selected examples from each task with explanations of the answer, task instructions, and control conditions (such as true non-explanatory statements). This small dataset contains these annotated examples; it is intended to be used in conjunction with the original tasks.


## Dataset creation process & limitations

A single author of the paper annotated this dataset, by reading the corresponding BIG-Bench problems and solutions and then attempting to write an explanation that would help a human to understand the answer. Although this process allowed for expert annotations on tasks where they would otherwise potentially be difficult to acquire (e.g. tasks that require understanding mathematical induction), it has corresponding limitations. Because of this time-intensive process, the dataset is small and does not include all BIG-Bench tasks. The selection criteria for tasks is described in further detail in the paper appendix. Furthermore, because a single author annotated the explanations, they may be biased, in the sense that they may include certain patterns of explanation (or even errors) that may or may not be beneficial.


## Contents

The folder contains three directories:
1. `untuned/`         This folder contains the raw, untuned explanations and control conditions, saved in a separate JSON file for each task.
2. `handtuned/`     This folder contains revised prompts that were created by tuning the explanations by hand from the best prompt, using the other examples in untuned/ as a validation set.
3. `selected/`         This folder contains revised prompts that were created by selecting which examples and explanations to include in the prompt, using the other examples in untuned/ as a validation set.


## Data structure

The examples in `untuned/` contain JSON structures which contain three items:
1. “examples_and_explanations”: an array containing three arrays each containing 5 items. These correspond to the three five-shot prompts created. The items in the lists are examples in the format of BIG-Bench (https://github.com/google/BIG-bench), except with additional fields containing “explanation”, “other_explanation”, “scrambled_explanation”, and “true_non_explanation”.
2. “zeroshot_descriptions”: An array containing three strings, each of which provides a distinct set of instructions for the task.
3. “zeroshot_nondescriptions”: An array containing three strings, each of which approximately matches the corresponding task instruction in length, but does not provide a task-specific instruction.

The files in `selected/` and `handtuned/` contain json objects containing complete prompts, keyed by their names.


## Code snippets

Code for loading the `untuned/` explanations and converting them into the prompt formats used in the paper can be found in `load_explanations.py`


## Citing this dataset

If you use these explanations, please cite both the associated paper and BIG-Bench:


Can language models learn from explanations in context? (https://arxiv.org/abs/2204.02329): 


@article{lampinen2022can,
  title={Can language models learn from explanations in context?},
  author={Lampinen, Andrew K and Dasgupta, Ishita and Chan, Stephanie CY and Matthewson, Kory and Tessler, Michael Henry and Creswell, Antonia and McClelland, James L and Wang, Jane X and Hill, Felix},
  journal={arXiv preprint arXiv:2204.02329},
  year={2022}
}



BIG-Bench (https://github.com/google/BIG-bench):


@article{bigbench,
   title = "Beyond the Imitation Game: Measuring and extrapolating the capabilities of language models",
   author = "{BIG-bench collaboration}", 
   year = "2021",
   journal = "In preparation",
   url = "https://github.com/google/BIG-bench/"
}
