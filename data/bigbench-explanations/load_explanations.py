# Copyright 2022 DeepMind Technologies Limited.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np

def create_prompts(exp_file_name, example_join='\n',
                   explanation_join='\nExplanation: ', instruction_join='\n',
                   force_task_prefix=False):
  with open(exp_file_name, 'r') as f:
    examples_and_explanations = json.load(f)
  prompts = {}
  prompts['none'] = ''
  if force_task_prefix:
    task_prefix = examples_and_explanations['task_prefix']

  ### examples no explanations or instruction
  for i, examples in enumerate(examples_and_explanations["fewshot_examples"]):
    this_prompt = ''
    if force_task_prefix:
      this_prompt += task_prefix
    for ex in examples:
      this_prompt += ex['input'] + ex['target'][0] + example_join
    prompts['fewshot_%i' % i] = this_prompt

  ### examples + explanations (or controls)
  for explain_condition in ['explanation', 'true_nonexplanation',
                            'scrambled_explanation', 'other_item_explanation']:
    for i, examples in enumerate(examples_and_explanations["fewshot_examples"]):
      this_prompt = ''
      if force_task_prefix:
        this_prompt += task_prefix
      for ex in examples:
        this_prompt += ex['input'] + ex['target'][0] + explanation_join
        this_prompt += ex[explain_condition] + example_join
      ec_str = explain_condition.replace("_", "")
      prompts['fewshot_%s_%i' % (ec_str, i)] = this_prompt

  ### instructions/noninstructions + maybe examples + maybe explanations
  for instr_cond in ["descriptions", "nondescriptions"]:
    ic = "instruction" if instr_cond == "descriptions" else "noninstruction"
    for i, (examples, instruction) in enumerate(zip(
        examples_and_explanations["fewshot_examples"],
        examples_and_explanations["zeroshot_%s" % instr_cond])):
      this_initial_prompt = instruction + instruction_join
      if force_task_prefix:
        this_initial_prompt += task_prefix
      prompts['%s_%i' % (ic, i)] = this_initial_prompt

      # examples + instructions
      this_prompt = this_initial_prompt
      for ex in examples:
        this_prompt += ex['input'] + ex['target'][0] + example_join
      prompts['%s_fewshot_%i' % (ic, i)] = this_prompt

      # examples + instructions + explanations
      this_prompt = this_initial_prompt
      for ex in examples:
        this_prompt += ex['input'] + ex['target'][0] + explanation_join
        this_prompt += ex['explanation'] + example_join
      prompts['%s_fewshot_explanation_%i' % (ic, i)] = this_prompt

  return prompts

if __name__ == "__main__":
  TASK_NAME = 'identify_odd_metaphor'
  exp_file_name = 'untuned/%s.json' % TASK_NAME
  prompts = create_prompts(exp_file_name, explanation_join='\nExplanation: ',
                           force_task_prefix=TASK_NAME == 'penguins_in_a_table')
  for k, v in prompts.items():
    print(k)
    print(v.__repr__())
