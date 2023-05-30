

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

# adapted from https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/peft-flan-t5-int8-summarization.ipynb
def prep_lora_model(
        model,
        train,
        args,
        target_modules = {"mt0": ["q", "v"], "bloom": ["query_key_value"], "t5": ["q", "v"], "mt5": ["q", "v"]}
):
    # see TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING ub peft/utils/other.py
    if "mt0" in args.model_id:
        target_modules = target_modules["mt0"]
    elif "bloom-" in args.model_id:
        target_modules = target_modules["bloom"]
    elif "mt5" in args.model_id:
        target_modules = target_modules["mt5"]
    elif "t5" in args.model_id:
        target_modules = target_modules["t5"]
    else:
        raise NotImplementedError(f"no target modules specified for {args.model_id}")
    
    if train is False and args.wandb_run is None:
        print('No finetuning or evaluation run specified, this is an ablation run, not using LoRA!')
        return model
    
    # Define LoRA Config 
    if train is True:
        print('Setting up LoRA!')
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=TaskType.CAUSAL_LM
        )
        # prepare int-8 model for training
        model = prepare_model_for_int8_training(model)

    else:
        print('Loading trained LoRA!')
        save_dir = os.path.join("trained_models", args.wandb_run)
        lora_config = LoraConfig.from_pretrained(save_dir)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # create method for generating text without teacher forcing but with gradients
    # this is achieved by removing the "no grad" decorator from the base_model of the peft

    # unwrapped = model.base_model.generate.__wrapped__
    # model.generate = MethodType(unwrapped, model.base_model)

    return model

def get_data_collator(model, tokenizer):
    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer,
    )
    return data_collator
    


def run_model(model, tokenizer, tokenized_dataset, args):

    train = args.train
    assert (train is True) or (train is False)

    model = prep_lora_model(model, train, args)
    data_collator = get_data_collator(model, tokenizer)

    # wandb settings
    if train is True and not args.debug:
        wandb.init(project="cot-instruction-tuning-v0", config=args)
        report_to = "wandb"
    else:
        wandb.init(mode="disabled") # Turned off wandb for evaluation
        report_to = None

    # metric = evaluate.load('accuracy')

    def compute_metrics(eval_preds):
        logits, labels = eval_preds # logits is a tuple of 2 tensors, we think first is prediction, second is last layer or smth    

        logits_argmax = np.argmax(logits, axis=2)
        assert logits_argmax.shape == labels.shape
        accuracy = np.logical_or(logits_argmax == labels, labels == -100).mean() #TODO: np.all now not yet, because it's hardcoded that label is len(1)

        # accuracy = np.all(np.logical_or(predictions == labels, labels == -100), axis=1).mean()

        # Compute samples_without_answer_fraction
        # Checking whether logits are zero, first for all words in vocab, then for all tokens in sequence
        samples_without_answer_mask = np.all(logits == 0, axis=(1,2))
        samples_without_answer = np.sum(samples_without_answer_mask)
        samples_without_answer_fraction = samples_without_answer / len(labels)
        return {"accuracy": accuracy, "invalid": samples_without_answer_fraction}

    output_dir="."
    # Define training args
    # training_args = Seq2SeqTrainingArguments(
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
                # TODO: Currently samples are being padded left and right, which is probably not good. See the warning:
                #   A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
                # TODO: Check if we need to add logs ourselves

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
                kwargs = {
                    "input_ids": inputs["input_ids"], 
                    "max_new_tokens": 32,
                }
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

                # sample_idx_without_answer = torch.arange(len(outputs), device=device)[start_of_answer_indices == -1]
                # for i in sample_idx_without_answer:
                #     print(f"No answer found in the {i}-ith sample:")
                #     input = tokenizer.decode(inputs["input_ids"][i])
                #     print(f"{input=}")
                #     # print(f"{outputs[i]=}")
                #     print(f"{tokenizer.decode(outputs[i])=}")


                # 2.3. Get outputs_without_answers as the outputs until the indices we have found
                outputs_without_answers = outputs.detach().clone()
                mask_hiding_answer_and_beyond = torch.arange(outputs.shape[1], device=device).repeat((batch_size,1)) > start_of_answer_indices[:, None]
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




    class AblationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Call forward!
            device = inputs["input_ids"].device
            labels = inputs["labels"]
            batch_size = inputs["input_ids"].shape[0]

            # print(inputs.keys())

            print(f"{inputs['input_ids'].shape=}")
            print(f"{inputs['labels'].shape=}")
            print(f"{inputs['attention_mask'].shape=}")

            forward_out = model.forward(inputs["input_ids"], labels=inputs["labels"], attention_mask=inputs['attention_mask'])

            # Compute logits for answers if necessary
            if return_outputs:
                # initialize answer_logits as a zeros tensor of shape (B, max_label_length, V)
                max_label_length = max([len(label) for label in labels])
                vocab_size = forward_out.logits.shape[-1]
                answer_logits = torch.zeros((batch_size, max_label_length, vocab_size), device=device)
                for i, label in enumerate(labels):
                    start_of_answer_content_index = 0
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

    if args.n_shot > 0 and not args.no_explanation:
        # Create Trainer instance
        print("Using CoTTrainer")
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics # commented because it doesn't work yet
        )
    else:
        print("Using AblationTrainer")
        trainer = AblationTrainer(
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