from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
import os
from datasets import load_dataset


def download_bigbench_drive(target_dir):
    """Download bigbench zipfile containing 4 tasks (all but truthful_qa) to target dir

    Args:
        target_dir (str): path to cache directory
    """
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
    download_bigbench_drive(target_dir)
    download_hf_datasets(target_dir)


def rename_keys(example):
    example["inputs"] = example["question"]
    example["targets"] = example["best_answer"]
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
        validation_path = Path(directory) / dataset_name / "train.jsonl"
        dataset = load_dataset(
            "json",
            data_files={"train": str(train_path), "validation": str(validation_path)},
        )
    elif dataset_name == "truthful_qa":
        dataset = load_dataset("truthful_qa", "generation")
        # truthful_qa only contains validation, so i renamed it to train for consistency
        print("truthful_qa only contains split validation, so i renamed it to train for consistency and removed validation")
        dataset["train"] = dataset["validation"]
        del dataset["validation"]
    else:
        raise NotImplementedError(f"no dataset available with name: {dataset_name}")

    if remap_names:
        dataset["train"].map(rename_keys).map(add_question_prompt)

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
