from datasets import load_from_disk
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb

# adapted from https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/peft-flan-t5-int8-summarization.ipynb
TARGET_MODULES = {"mt0": ["q", "v"], "bloom": ["query_key_value"], "t5": ["q", "v"], "mt5": ["q", "v"]}

def train_model(model, tokenizer, tokenized_dataset, args):
    # see TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING ub peft/utils/other.py
    if "mt0" in args.model_id:
        target_modules = TARGET_MODULES["mt0"]
    elif "bloom-" in args.model_id:
        target_modules = TARGET_MODULES["bloom"]
    elif "mt5" in args.model_id:
        target_modules = TARGET_MODULES["mt5"]
    elif "t5" in args.model_id:
        target_modules = TARGET_MODULES["t5"]
    else:
        raise NotImplementedError(f"no target modules specified for {args.model_id}")

    # Define LoRA Config 
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

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
    wandb.init(project="cot-instruction-tuning-v0", config=args)

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
        report_to="wandb",
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


    run_name = wandb.run.name

    # train model
    trainer.train()
    trainer.evaluate()
    trainer.save_model(os.path.join("trained_models", run_name))