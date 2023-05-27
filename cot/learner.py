from datasets import load_from_disk
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb
import evaluate
import numpy as np
import torch
from torch import nn
from undecorated import undecorated
from types import MethodType
from transform_outputs import transform_outputs

from transform_outputs import transform_outputs

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
    
    # Define LoRA Config 
    if train is True:
        print('Setting up LoRA!')
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=TaskType.SEQ_2_SEQ_LM
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
    generate_with_grad = undecorated(model.base_model.generate)
    model.base_model.generate = MethodType(generate_with_grad, model)

    return model

def get_data_collator(model, tokenizer):
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = 0 # T5 pad token is 0
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    return data_collator
    


def run_model(model, tokenizer, tokenized_dataset, args):

    # We are going to train if no wandb run is specified, otherwise we are going to evaluate that wandb run
    train = True if args.wandb_run is None else False

    model = prep_lora_model(model, train, args)
    data_collator = get_data_collator(model, tokenizer)

    # wandb settings
    # if train is True:
    #     wandb.init(project="cot-instruction-tuning-v0", config=args)
    #     report_to = "wandb"
    # else:
    wandb.init(mode="disabled") # Turned off wandb for evaluation
    report_to = None

    # metric = evaluate.load('accuracy')

    def compute_metrics(eval_preds):
        logits, labels = eval_preds # logits is a tuple of 2 tensors, we think first is prediction, second is last layer or smth    

        # print('len(logits)', len(logits))
        # print('logits[0]', logits[0].shape)
        # print('logits[1]', logits[1].shape)
        # print('labels', labels.shape)

        predictions = np.argmax(logits[0], axis=2).astype(np.int32)

        # Remove the explanation and only retain the answer
        predictions = transform_outputs(predictions)

        # print('predictions', predictions.shape)
        # print('labels', labels.shape)

        # print('predictions[0]', predictions[0])
        # print('labels[0]', labels[0])

        # predictions[-1] = labels[-1] # set last prediction to label only for testing!!!
        # labels[0] = np.array([-100] * labels.shape[1]) # set first label to -100 only for testing!!!

        accuracy = np.all(np.logical_or(predictions == labels, labels == -100), axis=1).mean()
        # print(np.all(predictions == labels, axis=1))
        # print('accuracy', accuracy)
        return {"accuracy": accuracy}

    output_dir="."
    # Define training args
    training_args = Seq2SeqTrainingArguments(
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
        full_determinism=True,
    )

    class CustomSeq2SeqTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            kwargs = {
                "input_ids": inputs["input_ids"], 
                "return_dict_in_generate": True, 
                "output_scores": True,
                "max_new_tokens": 250,
            }
            
            outputs = model.generate(**kwargs)
            logits = torch.concatenate(outputs.scores)
            print(outputs.keys())
            
            # Prints for debugging and checking if the output looks good
            # print(f"{inputs=}")
            # print(f"{inputs['input_ids'][0]=}")
            # print(f"{inputs['labels'][0]=}")
            print(f"{tokenizer.decode(inputs['input_ids'][0])=}")
            print(f"{tokenizer.decode(inputs['labels'][0])=}")
            # print(f"{outputs=}")
            print(f"{outputs.sequences=}")
            sample_output_decoded = tokenizer.decode(outputs.sequences[0])
            print(f"{sample_output_decoded=}")

            # Approach: 
            # Decode all the samples
            # find the position of the answer
            # throw away all logits except the answer logits
            # Throw away all logits longer than the label
            # Compute loss by computing crossentropy of the logits that are left, and the corresponding label 
            # Return loss
            
            # The code below may or may not be helpful for that, but 
            # for now, to enable debugging, we just return something that's backpropable
            return logits.mean()
        
            P = tokenizer.pad_token_id
            token_id_A = tokenizer.convert_tokens_to_ids("a")
            print(tokenizer.convert_tokens_to_ids("A"))

            print('token id A:', token_id_A)
            print('P', P)

            to_tokenize = outputs.sequences[0]
            print(tokenizer.decode(to_tokenize))

            answer_seqs, answer_indexes = transform_outputs(outputs.sequences, A_seq=token_id_A, P = P, return_indices=True)

            print(answer_indexes)

            # If no answer was found, we dismiss this sample by returning 0 loss
            # TODO: Compute crossentropy (below code is buggy)


            # Ignore samples for which no answer was found 
            valid_samples_mask = torch.tensor(answer_indexes != -1)
            for i in torch.arange(logits.shape[0])[not valid_samples_mask]:
                print(f"Dismissing the sample because no answer could be extracted from {answer_seqs[i]=}")

            logits = logits[valid_samples_mask]
            labels = inputs["labels"][valid_samples_mask]

            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Create Trainer instance
    trainer = CustomSeq2SeqTrainer(
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
