from datasets import load_from_disk
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb

# adapted from https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/peft-flan-t5-int8-summarization.ipynb

def train_model(model, tokenizer, tokenized_dataset, args):

    # Define LoRA Config 
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "v"],
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
        logging_strategy="steps",
        logging_steps=500,
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


    # train model
    trainer.train()