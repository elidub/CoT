import numpy as np

def compute_ablation_metrics(eval_preds):

    preds, labels = eval_preds

    assert preds.shape == labels.shape
    accuracy = np.all(np.logical_or(preds == labels, labels == -100), 1).mean()

    return {"accuracy": accuracy}


def qea_compute_metrics(eval_preds):
    preds, labels = eval_preds # logits is a tuple of 2 tensors, we think first is prediction, second is last layer or smth    

    # check elementwise equal, padding should also match
    elementwise = np.equal(preds,labels)

    # sum them per sample and check if all matched per row
    per_sample = np.sum(elementwise, axis=1)
    n_correct = per_sample == elementwise.shape[1]
    n_correct = np.sum(n_correct)
    accuracy = n_correct / labels.shape[0]

    # check how many rows were all pad
    n_padding_per_sample = preds == -100
    per_sample = np.sum(n_padding_per_sample, axis=1)
    n_no_answers = np.sum(per_sample == 3)
    return {"accuracy": accuracy, "invalid": n_no_answers}


def qae_compute_metrics(eval_preds):
    preds, labels = eval_preds # logits is a tuple of 2 tensors, we think first is prediction, second is last layer or smth    

    # check elementwise equal, padding should also match
    elementwise = np.equal(preds,labels)

    # sum them per sample and check if all matched per row
    per_sample = np.sum(elementwise, axis=1)
    n_correct = per_sample == elementwise.shape[1]
    n_correct = np.sum(n_correct)
    accuracy = n_correct / labels.shape[0]

    # check how many rows were all pad
    n_padding_per_sample = preds == -100
    per_sample = np.sum(n_padding_per_sample, axis=1)
    n_no_answers = np.sum(per_sample == 3)
    return {"accuracy": accuracy, "invalid": n_no_answers}