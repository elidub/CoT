from cot.model_utils import load_model_dicts, load_model
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import argparse
from datasets import load_from_disk


# The first two functions (data and train) are copied from the cellblocks from this notebook:
# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/peft-flan-t5-int8-summarization.ipynb

# See the two functions below (parse_option and main) to control this notebook



def data(model, tokenizer):

    from datasets import load_dataset

    # Load dataset from the hub
    dataset = load_dataset("samsum")

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")

    from datasets import concatenate_datasets
    import numpy as np
    # The maximum total input sequence length after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    # take 85 percentile of max length for better utilization
    max_source_length = int(np.percentile(input_lenghts, 85))
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # take 90 percentile of max length for better utilization
    max_target_length = int(np.percentile(target_lenghts, 90))
    print(f"Max target length: {max_target_length}")





    def preprocess_function(sample,padding="max_length"):
        # add prefix to the input for t5
        inputs = ["summarize: " + item for item in sample["dialogue"]]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # save datasets to disk for later easy loading
    tokenized_dataset["train"].save_to_disk("data/train")
    tokenized_dataset["test"].save_to_disk("data/eval")

def train(model, tokenizer):
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





def parse_option():
    parser = argparse.ArgumentParser(description="Testing various models")
    parser.add_argument('--model_type', default = 'google/flan-t5-small', type=str, help='Model type')
    args = parser.parse_args()
    return args

def main(args):
    m_id = args.model_type

    m_dicts = load_model_dicts()
    model, tokenizer = load_model(m_id, m_dicts[m_id])

    data(model, tokenizer)
    train(model, tokenizer)


if __name__ == "__main__":
    args = parse_option()
    print(args)
    main(args)