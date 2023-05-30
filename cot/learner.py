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

import sys
sys.path.insert(1, sys.path[0] + '/../')

from cot.transform_outputs import transform_outputs
from cot.model_utils import prep_lora_model
from cot.metrics import custom_compute_metrics, compute_ablation_metrics
from cot.trainer import AblationTrainer

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
                labels = inputs["labels"]
                a_seq = torch.tensor([3000], device=device)           # "A:"
                # a_seq = torch.tensor([3000, 210], device=inputs["input_ids"].device)      # "A: "
                batch_size = inputs["input_ids"].shape[0]

                # 1. Generate outputs without grad
                # We do beam search with repetition penalty to avoid the model getting stuck repeating itself.
                repetition_penalty = 2.0
                num_beams = 3
                kwargs = {
                    "input_ids": inputs["input_ids"], 
                    "max_new_tokens": 32,

                    # Beam search params
                    "do_sample": True,
                    "num_beams": num_beams,
                    "repetition_penalty": repetition_penalty,
                }
                print(f"{num_beams=}")
                print(f"{repetition_penalty=}")
                outputs = model.generate(**kwargs)

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



                sample_idx_without_answer = torch.arange(len(outputs), device=device)[start_of_answer_indices == -1]
                for i in sample_idx_without_answer:
                    # print(f"No answer found in the {i}-ith sample:")
                    # input = tokenizer.decode(inputs["input_ids"][i])
                    # print(f"{input=}")
                    # print(f"{outputs[i]=}")
                    # print(f"{outputs[i]=}")
                    # print(f"{tokenizer.decode(outputs[i])=}")

                    # If there's no a_seq, but the sequence ends in a </s> with enough free space in the end, we replace the </s> token with the a_seq
                    # Example motivating this: tokenizer.decode(outputs[i])='<pad><pad><pad>Q:What is 842 divided by 1?\n  choice: 456\n  choice: 842\n  choice: house\n  choice: 14513\n  choice: 1\n  choice: banana\n  choice: 820\nExplanation:Dividing any number by 1 does not change it, so 842 / 1 = 842.\nA:842\nQ:What is 355 divided by 71?\n  choice: 5\n  choice: banana\n  choice: 807\n  choice: 7\n  choice: house\n  choice: 6\nExplanation:355 is close to 400, and 71 is close to 80, and 400 / 80 = 5, so we can guess that the answer must be 5.\nA:5\nQ:What is 705 divided by 5?\n  choice: 445\n  choice: 113\n  choice: 141\n  choice: banana\n  choice: house\n  choice: 73442\n  choice: 2\nExplanation:705 / 5 = (700 / 5) + (5 / 5) = 140 + 1 = 141.\nA:141\nQ: What is 321 divided by 3?\n  choice: 107\n  choice: 12944\n  choice: banana\n  choice: 6\n  choice: 852\n  choice: 759\n  choice: house\nExplanation:321 / 3 = (320 / 3) + (3 / 3) = 108 + 1 = 109.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
                    eos_indices = (outputs[i] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_indices) > 0:
                        eos_idx = eos_indices[0].item()
                        if eos_idx + len(a_seq) + labels.shape[1] < outputs.shape[1]:
                            start_of_answer_indices[i] = eos_idx
                            outputs[i, eos_idx:eos_idx + len(a_seq)] = a_seq
                            # print(f"{tokenizer.decode(outputs[i])=}")
                            continue
                    print(f"No answer found in the {i}-ith sample and it could not be fixed:")
                    print(f"{tokenizer.decode(outputs[i])=}")

                # Answer sequences at the very end, when there's not enough space for the real answer, do not count
                start_of_answer_indices[start_of_answer_indices >= outputs_without_inputs.shape[-1] - labels.shape[1]] = -1

                # 2.3. Get outputs_without_answers as the outputs until the indices we have found
                tmp_start_of_answer_indices = start_of_answer_indices.clone()
                tmp_start_of_answer_indices[tmp_start_of_answer_indices == -1] = 100000 # hide nothing if label is invalid
                mask_hiding_answer_and_beyond = (torch.arange(outputs.shape[1], device=device).repeat((batch_size, 1)) >= tmp_start_of_answer_indices[:, None] + len(a_seq))

                outputs_without_answers = outputs.detach().clone()
                outputs_without_answers[mask_hiding_answer_and_beyond] = tokenizer.pad_token_id

                # Block of debug prints
                # print(f"{tokenizer.decode(inputs['input_ids'][0])=}")
                # print(f"{tokenizer.decode(outputs[0])=}")
                # print(f"{tokenizer.decode(outputs_without_inputs[0])=}")
                # print(f"{tokenizer.decode(outputs_without_answers[0])=}") # ends in "...\nA:<pad><pad>...<pad>"

                # 3. TODO: Check if nan for loss is OK for cases when no a_seq is found (we want to dismiss the sample/batch then)

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

                    # # Prints to confirm that labels and inputs are aligned (confirmed)
                    # print(f"{tokenizer.decode(full_correct_sequences[i][start_of_answer_index-len(a_seq) : start_of_answer_index+len(label)])=}")
                    # print(f"{tokenizer.decode(labels_formatted[i][start_of_answer_index-len(a_seq) : start_of_answer_index+len(label)])=}")


                # Create attention_mask with 0s for padding tokens and the label, as the model should not attend to them
                attention_mask = torch.ones_like(full_correct_sequences)
                attention_mask[outputs_without_answers == tokenizer.pad_token_id] = 0

                # for i in range(len(labels)):
                #     print(f"{tokenizer.decode(full_correct_sequences[i])=}")

            # Call forward!
            forward_out = model.forward(full_correct_sequences, labels=labels_formatted, attention_mask=attention_mask)

            if (start_of_answer_indices == -1).any():
                print(f"No answer found at least once in this batch.")
            if (start_of_answer_indices == -1).all():
                print(f"No answer found anywhere in this batch.")

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

            print(f"{forward_out.loss=}")
            return (forward_out.loss, answer_logits) if return_outputs else forward_out.loss



            # # # Approach 2: Without labels in forward pass (surely no teacher-forcing)
            # forward_out = model.forward(outputs_without_answers, attention_mask=mask_hiding_answer_and_beyond)
            # print("Approach 2:")
            # print(f"{forward_out.keys()=}")
            # print(f"{padded_outputs_without_answers.shape=}")
            # print(f"{forward_out.logits.shape=}")


            # print(f"{inputs['attention_mask']=}")
            # print(f"{outputs.shape=}")
            # print(f"{outputs[0]=}")
            # print(f"{outputs[inputs['attention_mask']].shape=}")
            # print(f"{outputs[inputs['attention_mask']][0]=}")
            # print(f"{tokenizer.decode(outputs[inputs['attention_mask']][0])=}")
            # END Alternative approach

            # kwargs = {
            #     "input_ids": inputs["input_ids"], 
            #     "return_dict_in_generate": True, 
            #     "output_scores": True,
            #     "max_new_tokens": 32,
            # }
            # outputs = model.generate(**kwargs)
            # print(outputs.keys())
            # logits = torch.stack(outputs.scores, dim=1)
            # vocab_size = logits.shape[-1]
            
            # # Prints for debugging and checking if the output looks good
            # # print(f"{inputs=}")
            # # print(f"{inputs['input_ids'][0]=}")
            # # print(f"{inputs['labels'][0]=}")
            # print(f"{tokenizer.decode(inputs['input_ids'][0])=}")
            # print(f"{tokenizer.decode(inputs['labels'][0])=}")
            
            # # get length from first input
            # input_length = len(inputs['input_ids'][0])
            # # print(f"{outputs=}")
            # # print(f"{outputs.sequences=}")
            # print(f"{tokenizer.decode(outputs.sequences[0])=}")
            # print(f"{logits.shape=}")
            # print(f"{logits}")
            # # A: is only 1 token it seems
            # a_seq = torch.tensor([3000], device=outputs.sequences.device)           # "A:"
            # # a_seq = torch.tensor([3000, 210], device=outputs.sequences.device)      # "A: "
            # answer_length = 2

            # generated_sequences = outputs.sequences[:, input_length:]
            
            # # try to find answer tokens in the input length
            # tokens_padded, start_of_answer_indices = transform_outputs(generated_sequences, a_seq, P=tokenizer.pad_token_id)
            
            # # print(f"{tokens_padded[0]=}")
            # print(f"{tokenizer.decode(tokens_padded[0])=}")
            # print(f"{start_of_answer_indices=}")

            # batch_size = len(inputs['input_ids'])
            # batch_answer = torch.zeros(batch_size, answer_length, device=outputs.sequences.device)
            # for i in range(batch_size):
            #     batch_answer[i] = torch.tensor([3000, inputs['labels'][i][-1]])
            # batch_answer = batch_answer.long()

            # batch_output = torch.zeros((batch_size, answer_length, vocab_size), device=outputs.sequences.device)
            
            # # if no answers were generated pass a all unknown batch (all zeros) is passed instead
            # if start_of_answer_indices.nelement() != 0:
            #     # otherwise we just get the 
            #     batch_output = torch.zeros(batch_size, answer_length, vocab_size, device=outputs.sequences.device)
            #     for i in range(batch_size):
            #         try:
            #             batch_output[i] = logits[i][start_of_answer_indices[i]:start_of_answer_indices[i] + answer_length]
                    
            #             print("found answer: ", tokenizer.decode(generated_sequences[i][start_of_answer_indices[i]:start_of_answer_indices[i] + answer_length]))
            #             print("actual answer: ", tokenizer.decode(batch_answer[i]))
            #         # if only some of the elements in the batch have an answer
            #         except RuntimeError:
            #             print(f"No answer found in ")
            #             pass


            # batch_output = batch_output.view(batch_size * answer_length, -1)
            # batch_answer = batch_answer.view(-1)
            # loss = nn.functional.cross_entropy(batch_output, batch_answer)
            # print("loss: ", loss)
            # print("-" * 50)
            # return loss
                
            # # print(f"{sample_output_decoded=}")

            # # Approach: 
            # # Decode all the samples
            # # find the position of the answer
            # # throw away all logits except the answer logits
            # # Throw away all logits longer than the label
            # # Compute loss by computing crossentropy of the logits that are left, and the corresponding label 
            # # Return loss
            
            # # The code below may or may not be helpful for that, but 
            # # for now, to enable debugging, we just return something that's backpropable
            # # return logits.mean()
        
            # P = tokenizer.pad_token_id
            # token_id_A = tokenizer.convert_tokens_to_ids("a")
            # print(tokenizer.convert_tokens_to_ids("A"))

            # print('token id A:', token_id_A)
            # print('P', P)

            # to_tokenize = outputs.sequences[0]
            # print(tokenizer.decode(to_tokenize))

            # answer_seqs, answer_indexes = transform_outputs(outputs.sequences, A_seq=token_id_A, P = P, return_indices=True)

            # print(answer_indexes)

            # # If no answer was found, we dismiss this sample by returning 0 loss
            # # TODO: Compute crossentropy (below code is buggy)


            # # Ignore samples for which no answer was found 
            # valid_samples_mask = torch.tensor(answer_indexes != -1)
            # for i in torch.arange(logits.shape[0])[not valid_samples_mask]:
            #     print(f"Dismissing the sample because no answer could be extracted from {answer_seqs[i]=}")

            # logits = logits[valid_samples_mask]
            # labels = inputs["labels"][valid_samples_mask]

            # loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            # loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            # return (loss, outputs) if return_outputs else loss


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
    eval = trainer.evaluate()

    if train is False:
        print(eval)
