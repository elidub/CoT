from datasets import load_from_disk
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb

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
    
    return model
    


def run_model(model, tokenizer, tokenized_dataset, args):

    train = True if args.wandb_run is None else False
    print(f'Train: {train}')


    model = prep_lora_model(model, train, args)

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    output_dir="."

    # wandb
    if train is True:
        wandb.init(project="cot-instruction-tuning-v0", config=args)
    report_to = "wandb" if train else None

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

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


    # Train model
    if train is True:
        print('Training!')
        trainer.train()

        # Save model
        save_dir = os.path.join("trained_models", wandb.run.name)
        model.save_pretrained(save_dir)



    # Evaluate model
    print('Evaluating!')
    trainer.evaluate()

    #     # save_dir = os.path.join("trained_models", wandb.run.name)
    #     model.save_pretrained('./lora_pretrained')
    #     trainer.save_model('./lora_pretrained_2')
    #     # trainer.save_model(save_dir)
        # trainer.model.config.to_json_file(os.path.join(save_dir, "config.json")) # This should be done automatically by trainer.save_model !?!?
