from utils import load_model_dicts, load_model
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

from datasets import load_from_disk


def run():



    m_dicts = load_model_dicts()

    m_id = 'google/flan-t5-base'
    model, tokenizer = load_model(m_id, m_dicts[m_id], hf_cache = '/nfs/scratch/atcs_cot/hf_cache_dev/')

    tokenized_dataset = {}
    tokenized_dataset['train'] = load_from_disk("data/train")

    # Define LoRA Config 
    lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
    )
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()



    from transformers import DataCollatorForSeq2Seq

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )



    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    output_dir="."

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
            auto_find_batch_size=True,
        learning_rate=1e-3, # higher learning rate
        num_train_epochs=5,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
        report_to="tensorboard",
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


    # train model
    trainer.train()


if __name__ == '__main__':
    run()