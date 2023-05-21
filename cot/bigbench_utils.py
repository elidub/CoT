from pathlib import Path
import os
from datasets import load_dataset


def download_bigbench_drive(target_dir):
    """Download bigbench zipfile containing 4 tasks (all but truthful_qa) to target dir

    Args:
        target_dir (str): path to cache directory
    """
    from google_drive_downloader import GoogleDriveDownloader as gdd
    # url to my google drive
    gdd.download_file_from_google_drive(file_id='1m6sIfgb7hUMMjOmeT1uJp',
                                    dest_path='./bigbench.zip',
                                    unzip=True)

    os.remove('./bigbench.zip')


def download_hf_datasets(target_dir):
    """Download datasets from the Huggingface Hub

    Args:
        target_dir (str): path to cache directory
    """
    load_dataset("truthful_qa", "generation", cache_dir=target_dir)


def download_datasets(target_dir):
    # download_bigbench_drive(target_dir)
    download_hf_datasets(target_dir)


def rename_keys(example):
    inputs_key = [key for key in example.keys() if key in ["inputs", "question"]][0]
    example["inputs"] = example[inputs_key]

    targets_key = [key for key in example.keys() if key in ["targets", "best_answer"]][0]
    example["targets"] = example[targets_key]
    return example


def add_question_prompt(example):
    example["inputs"] = "Q: " + example["inputs"]
    return example


def load_bigbench_dataset(dataset_name, directory, remap_names=True):
    if dataset_name in [
        "modified_arithmetic",
        "disambiguation_qa",
        "odd_one_out",
        "presuppositions_as_nli",
    ]:
        train_path = Path(directory) / dataset_name / "train.jsonl"
        validation_path = Path(directory) / dataset_name / "validation.jsonl"
        dataset = load_dataset(
            "json",
            data_files={"train": str(train_path), "validation": str(validation_path)},
        )
    elif dataset_name == "truthful_qa":
        dataset = load_dataset("truthful_qa", "generation")
        # truthful_qa only contains validation, so i renamed it to train for consistency
        print("truthful_qa only contains split validation, use the validation as train and split it")
        # split with fixed seed so the split should always be the same. 
        # COT dataset is called twice so the dataset is constructed twice, but if the seed stays the same that's fine
        dataset = dataset['validation'].train_test_split(test_size=0.3, seed=42)
        # rename test to validation
        dataset['validation'] = dataset.pop('test')

    else:
        raise NotImplementedError(f"no dataset available with name: {dataset_name}")

    if remap_names:
        dataset["train"] = dataset["train"].map(rename_keys).map(add_question_prompt)
        try:
            dataset["validation"] = dataset["validation"].map(rename_keys).map(add_question_prompt)
        except KeyError:
            pass

    return dataset


if __name__ == "__main__":
    cache_dir = "/tmp/bigbench"
    # download datsets like this:
    download_datasets(cache_dir)

    # load datasets like this:
    dataset_truthful_qa = load_bigbench_dataset("truthful_qa", cache_dir)
    dataset_modified_arithmetic = load_bigbench_dataset(
        "modified_arithmetic", cache_dir
    )
    dataset_disambiguation_qa = load_bigbench_dataset("disambiguation_qa", cache_dir)
    dataset_odd_one_out = load_bigbench_dataset("odd_one_out", cache_dir)
    dataset_presuppositions_as_nli = load_bigbench_dataset(
        "presuppositions_as_nli", cache_dir
    )
