from datasets import load_from_disk
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainingArguments, DataCollator, DataCollatorWithPadding, DataCollatorForLanguageModeling
import wandb
import evaluate
import numpy as np
import torch
from torch import nn
from undecorated import undecorated
from types import MethodType
import wandb

import sys
sys.path.insert(1, sys.path[0] + '/../')

from cot.transform_outputs import transform_outputs
from cot.model_utils import prep_lora_model
from cot.metrics import custom_compute_metrics, compute_ablation_metrics
from cot.trainer import AblationTrainer
from cot.utils import update_results

def get_data_collator(model, tokenizer):
    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer,
    )
    return data_collator
    


def run_model(model, tokenizer, tokenized_dataset, args):

    train = args.train

    model = prep_lora_model(model, train, args)
    data_collator = get_data_collator(model, tokenizer)

    # wandb settings
    if train is True and not args.debug:
        wandb.init(project="cot-instruction-tuning-v0", config=args)
        report_to = "wandb"
    else:
        wandb.init(mode="disabled") # Turned off wandb for evaluation
        report_to = None

    output_dir="."
    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, # higher learning rate
        num_train_epochs=args.max_epochs,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to=report_to,
        seed=args.seed,
        full_determinism=False, # otherwise the forward pass gives RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation
        dataloader_drop_last=True # Otherwise eval crashes on the last iteration requiring 28TB Memory
    )

    # class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    class CustomSeq2SeqTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            with torch.no_grad():
                # Alternative approach
                # Advantages:
                # - Less memory used 
                #       with logits we crash at: --model_id bigscience/bloom-560m --batch_size 2 --n_shot 4, "max_new_tokens": 32
                #       without logits we can generate: --model_id bigscience/bloom-1b1 --batch_size 4 --n_shot 4, "max_new_tokens": 32
                # - We might get teacher-forcing for the answer (maybe not needed)
                # - We don't need to check for different answer lengths manually (tokenizer(contradiction) is [7337, 326, 8156])
                # - Guarantee that there's no funky shit going on with cumulative gradients, where outputs of the explanation affect logits of the answer
                # 0.: Some useful variables for later on
                device = inputs["input_ids"].device
                debug = True
                log_excessively = True
                log_prefix = "train_" if self.is_in_train else "val_"
                labels = inputs["labels"]
                a_seq = torch.tensor([3000], device=device)           # "A:"
                # a_seq = torch.tensor([3000, 210], device=inputs["input_ids"].device)      # "A: "
                batch_size = inputs["input_ids"].shape[0]

                # 1. Generate outputs without grad
                kwargs = {
                    "input_ids": inputs["input_ids"], 
                    "max_new_tokens": 50,
                }
                # If specified, we do beam search with repetition penalty to avoid the model getting stuck repeating itself.
                if args.num_beams:
                    kwargs["do_sample"] = True
                    kwargs["num_beams"] = args.num_beams
                    kwargs["repetition_penalty"] = 2.0
                    print(f"Generating with beams using {kwargs}")
                outputs = model.generate(**kwargs)

                if log_excessively and self.is_in_train:
                    unique_tokens_per_sample = len(outputs.unique()) / batch_size
                    wandb.log({log_prefix + "unique_tokens_per_sample" : unique_tokens_per_sample}, commit=False)

                # 2. Compute input_with_explanations as the entire output where the answer itself is replaced by padding tokens
                # 2.1. Generate mask_hiding_non_inputs
                mask_hiding_non_inputs = inputs['attention_mask'].bool() # inputs['attention_mask'] hides all non-input tokens
                # Now the mask needs to be extended to the output size with True values, as the new output should not be hidden
                mask_hiding_non_inputs_extension = torch.zeros((batch_size, outputs.shape[1] - mask_hiding_non_inputs.shape[1]), dtype=torch.bool, device = device)
                mask_hiding_non_inputs = torch.concatenate((mask_hiding_non_inputs, mask_hiding_non_inputs_extension), axis=1)

                # 2.2. Get indices in the output where the model's answer starts
                # Overwrite all the inputs with padding tokens, so we do not find answer tokens from the support in the next step
                outputs_without_inputs = outputs.detach().clone()
                outputs_without_inputs[mask_hiding_non_inputs] = tokenizer.pad_token_id
                # Find the indices of the first answer token in the output generated by the model
                _, start_of_answer_indices = transform_outputs(outputs_without_inputs, a_seq, P=tokenizer.pad_token_id)


                if debug: 
                    print("=" * 20 + " summary of batch results " + "=" * 20)
                    print(f"{args.num_beams=}")

                sample_idx_without_answer = torch.arange(len(outputs), device=device)[start_of_answer_indices == -1]
                n_samples_fixed_by_eos_replacement = 0
                for i in sample_idx_without_answer:
                    print(f"No answer found in the {i}-ith sample")
                    if args.allow_answer_at_eos:
                        print("Looking for possible replacements..")
                        # If there's no a_seq, but the sequence ends in a </s> with enough free space in the end, we replace the </s> token with the a_seq
                        # Example motivating this: tokenizer.decode(outputs[i])='<pad><pad><pad>Q:What is 842 divided by 1?\n  choice: 456\n  choice: 842\n  choice: house\n  choice: 14513\n  choice: 1\n  choice: banana\n  choice: 820\nExplanation:Dividing any number by 1 does not change it, so 842 / 1 = 842.\nA:842\nQ:What is 355 divided by 71?\n  choice: 5\n  choice: banana\n  choice: 807\n  choice: 7\n  choice: house\n  choice: 6\nExplanation:355 is close to 400, and 71 is close to 80, and 400 / 80 = 5, so we can guess that the answer must be 5.\nA:5\nQ:What is 705 divided by 5?\n  choice: 445\n  choice: 113\n  choice: 141\n  choice: banana\n  choice: house\n  choice: 73442\n  choice: 2\nExplanation:705 / 5 = (700 / 5) + (5 / 5) = 140 + 1 = 141.\nA:141\nQ: What is 321 divided by 3?\n  choice: 107\n  choice: 12944\n  choice: banana\n  choice: 6\n  choice: 852\n  choice: 759\n  choice: house\nExplanation:321 / 3 = (320 / 3) + (3 / 3) = 108 + 1 = 109.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
                        eos_indices = (outputs[i] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                        if len(eos_indices) > 0:
                            eos_idx = eos_indices[0].item()
                            if eos_idx + len(a_seq) + labels.shape[1] < outputs.shape[1]:
                                start_of_answer_indices[i] = eos_idx
                                outputs[i, eos_idx:eos_idx + len(a_seq)] = a_seq
                                n_samples_fixed_by_eos_replacement += 1
                                # print(f"{tokenizer.decode(outputs[i])=}")
                                continue
                            print(f"No answer found in the {i}-ith sample and it could not be fixed:")
                            print(f"{tokenizer.decode(outputs[i])=}")
                if log_excessively:
                    wandb.log({log_prefix + "n_samples_fixed_by_eos_replacement" : n_samples_fixed_by_eos_replacement}, commit=False)

                # Answer sequences at the very end, when there's not enough space for the real answer, do not count
                start_of_answer_indices[start_of_answer_indices >= outputs_without_inputs.shape[-1] - labels.shape[1]] = -1

                # 2.3. Get outputs_without_answers as the outputs until the indices we have found
                tmp_start_of_answer_indices = start_of_answer_indices.clone()
                tmp_start_of_answer_indices[tmp_start_of_answer_indices == -1] = 100000 # hide nothing if label is invalid
                mask_hiding_answer_and_beyond = (torch.arange(outputs.shape[1], device=device).repeat((batch_size, 1)) >= tmp_start_of_answer_indices[:, None] + len(a_seq))

                outputs_without_answers = outputs.detach().clone()
                outputs_without_answers[mask_hiding_answer_and_beyond] = tokenizer.pad_token_id

                if log_excessively:
                    mask_hiding_everything_but_explanation = torch.logical_and(mask_hiding_answer_and_beyond, ~mask_hiding_non_inputs)
                    average_explanation_length = mask_hiding_everything_but_explanation.sum() / batch_size
                    wandb.log({log_prefix + "average_explanation_length" : average_explanation_length}, commit=True) # Commit as this is the last call of log

                # Block of debug prints
                n_samples_contained_answer = torch.sum(start_of_answer_indices != -1)
                if debug:
                    try:
                        print(f"{n_samples_contained_answer} / {batch_size}, samples contained an answer!")
                        print("")
                        print(f"input for the model:\n ", repr(tokenizer.decode(inputs["input_ids"][0])))  # ends in "...\nA:<pad><pad>...<pad>"
                        print("-" * 50)
                        print(
                            f"generated answer (sample=0):\n", repr(tokenizer.decode(outputs_without_inputs[0][outputs_without_inputs[0] != 3]))
                        )
                        print(
                            f"generated answer (sample=1):\n", repr(tokenizer.decode(outputs_without_inputs[1][outputs_without_inputs[0] != 3]))
                        )
                        print("-" * 50)
                    except:
                        pass

                labels_for_decoding = inputs["labels"].clone()
                actual_answer = tokenizer.decode(labels_for_decoding[0][labels_for_decoding[0] != -100])
                if not hasattr(self, "accuracies"):
                    self.accuracies = []

                batch_accuracies = []
                for b in range(batch_size):

                    if start_of_answer_indices[b] != -1:
                        generated_answer = outputs[
                            b, start_of_answer_indices[b]: start_of_answer_indices[b] + len(a_seq) + labels.shape[1]
                        ]
                        if len(generated_answer) == 0:
                            acc = 0
                        else:
                            unpadded_label = inputs["labels"][b][inputs["labels"][b] != -100]
                            # print(f"predicted answer: {tokenizer.decode(generated_answer[1:unpadded_label.shape[0]+1])}")
                            # skip answer token
                            pred = generated_answer[1:unpadded_label.shape[0]+1]
                            if debug:
                                print(f"target: {tokenizer.decode(unpadded_label)}, generated (idx={start_of_answer_indices[b]}): {tokenizer.decode(pred[:unpadded_label.shape[0]])}")
                            acc = int(torch.equal(unpadded_label, pred))
                        batch_accuracies.append(acc)
                    else:
                        batch_accuracies.append(0)

                    batch_acc = sum(batch_accuracies) / batch_size
                    self.accuracies.append(batch_acc)
                else:
                    print(f"no answer token predicted at index {b}!")

                    print(f"actual answer:    {actual_answer}")
                    n_running_mean_accuracy = 20
                    print(
                        f"average accuracy of the last {n_running_mean_accuracy} batches: {sum(self.accuracies[-n_running_mean_accuracy:]) / n_running_mean_accuracy}"
                    )
                    print("-" * 50)

                # 4. Call model.forward()
                # See https://huggingface.co/docs/transformers/model_doc/bloom#transformers.BloomForCausalLM.forward
                # and https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/bloom/modeling_bloom.py#L881
                # Create full_correct_sequence by adding the labels in the correct places, and
                # Create labels_formatted in the same shape, but everything except the label is -100 because we do not want loss for those tokens 
                full_correct_sequences = outputs_without_answers
                labels_formatted = torch.zeros_like(full_correct_sequences) - 100
                for i, label in enumerate(labels):
                    label_unpadded = label[label != -100]
                    start_of_answer_index = start_of_answer_indices[i] + len(a_seq)
                    full_correct_sequences[i, start_of_answer_index:start_of_answer_index+len(label_unpadded)] = label_unpadded
                    

                    if start_of_answer_indices[i] == -1:
                        # If no answer was found, we ignore the entire sample by providing a label with -100 everywhere
                        continue
                    else:
                        # Give loss for the "A:" part
                        labels_formatted[i][start_of_answer_index-len(a_seq): start_of_answer_index] = a_seq
                        # Give loss for the correct label
                        labels_formatted[i, start_of_answer_index:start_of_answer_index+len(label_unpadded)] = label_unpadded

                        # the model will also use the generated cot for loss if the answer was correct.
                        if args.reward_succesful_explanations and batch_accuracies[i] == 1:
                            if debug:
                                print(f"model will approve of this explanation (sample={i})")
                            start_from = inputs['input_ids'][0].shape[0]
                            until = start_of_answer_index  # approve also of everything up to start index
                            print(fr"'{tokenizer.decode(full_correct_sequences[i, start_from:until])}'")
                            labels_formatted[i, start_from:until] = full_correct_sequences[i, start_from:until]


                    # # Prints to confirm that labels and inputs are aligned (confirmed)

                # Create attention_mask with 0s for padding tokens and the label, as the model should not attend to them
                attention_mask = torch.ones_like(full_correct_sequences)
                attention_mask[outputs_without_answers == tokenizer.pad_token_id] = 0

                 # ================= answer loss ========================
                # for each element in the batch that did not predict a label, we pass as label
                # the answer token + the actual label at the end of the context
                desired_answer_index = outputs_without_inputs[0].shape[-1] -1 - labels.shape[1] - a_seq.shape[0]
                desired_label_index = desired_answer_index + a_seq.shape[0]

                # adding it for any sample in a batch if its missing results in the 
                # model learning that it just can output A:<answer>, only use it as a last resort
                if n_samples_contained_answer == 0:
                    if debug:
                        print("answer loss triggered!")
                    for i, soa_idx in enumerate(start_of_answer_indices):
                        if soa_idx == -1:
                            label = labels[i]
                            label_unpadded = label[label != -100]
                            if debug:
                                print(f"sample {i} from batch received label {tokenizer.decode(label_unpadded)}")
                            labels_formatted[i][desired_answer_index: desired_answer_index + a_seq.shape[0]] = a_seq
                            labels_formatted[i, desired_label_index:desired_label_index+len(label_unpadded)] = label_unpadded

            # Call forward!
            forward_out = model.forward(full_correct_sequences, labels=labels_formatted, attention_mask=attention_mask)

            if (start_of_answer_indices == -1).any():
                print(f"At least one sample was missing an answer in this batch.")
            if (start_of_answer_indices == -1).all():
                print(f"No answer found anywhere in this batch.")
                # returning nan skips this skep, see line 1936 in transformers/trainer.py (inner loop)
                forward_out.loss = torch.tensor([torch.nan], device=device)
                # forward_out.loss = torch.log(forward_out.loss + 1)

            # Compute logits for answers if necessary
            if return_outputs:
                # initialize answer_logits as a zeros tensor of shape (B, max_label_length, V)
                max_label_length = max([len(label) for label in labels])
                vocab_size = forward_out.logits.shape[-1]
                answer_logits = torch.zeros((batch_size, max_label_length, vocab_size), device=device)
                for i, label in enumerate(labels):
                    if start_of_answer_indices[i] == -1:
                        # If no answer was found, we leave the answer_logits as zeros.
                        # This indicates that no answer was given by the model at all.
                        continue
                    else:
                        start_of_answer_content_index = 0 if start_of_answer_indices[i] == -1 else start_of_answer_indices[i] + len(a_seq)
                        # Determine sample_answer_logits for this sample
                        sample_answer_logits = forward_out.logits[i][start_of_answer_content_index : start_of_answer_content_index+len(label)]

                        # Print confirming that we have sliced the correct indices
                        # print("The following is only the answer content: ")
                        # print(f"{tokenizer.decode(outputs[i][start_of_answer_content_index : start_of_answer_content_index+len(label)])}")  # prints e.g. "57" for arithmetic

                        # Fill in this sample's logits into answer_logits.
                        # If this label is shorter than max_label_length, leave zeros for the remaining tokens unchanged.
                        # These zeros will later be ignored because the label is -100 there.
                        answer_logits[i,:len(label), :] = sample_answer_logits

                answer_logits = torch.cat((torch.zeros_like(answer_logits[0]).unsqueeze(0), answer_logits)) # For this https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/trainer.py#L3526

            print(f"loss: {forward_out.loss}")
            return (forward_out.loss, answer_logits) if return_outputs else forward_out.loss

    if args.n_shot > 0 and not args.no_explanation:
        print("Using CoTTrainer")
        CustomTrainer = CustomSeq2SeqTrainer
        compute_metrics = custom_compute_metrics
    else:
        print("Using AblationTrainer")
        CustomTrainer = AblationTrainer
        compute_metrics = compute_ablation_metrics

    # Create Trainer instance
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics # commented because it doesn't work yet
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if train is True:
        # Train model
        print('Training!')
        trainer.train()

        # Save model
        save_dir = os.path.join("trained_models", wandb.run.name)
        model.save_pretrained(save_dir)

    # Evaluate model
    print('Evaluating!')
    evals = trainer.evaluate()

    if train is False:
        print(evals)

    # update results
    update_results(results = 'store/results.json', evals = evals, args = args)
