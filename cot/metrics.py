import numpy as np

def compute_ablation_metrics(eval_preds):

    preds, labels = eval_preds

    assert preds.shape == labels.shape
    accuracy = np.all(np.logical_or(preds == labels, labels == -100), 1).mean()

    return {"accuracy": accuracy}


def custom_compute_metrics(eval_preds):
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