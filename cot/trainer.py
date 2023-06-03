from transformers import Trainer
import torch
from types import MethodType

import sys
sys.path.insert(1, sys.path[0] + '/../')

class AblationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Call forward!
        # device = inputs["input_ids"].device
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        # batch_size = inputs["input_ids"].shape[0]

        # Generate without gradients
        outputs = model.generate(input_ids = input_ids, max_new_tokens = 32)

        # We are interested in the labels that don't have ignore index
        labels_unpadded = [labels != -100][0]

        # The output is evertyhing the that comes after the input indices
        outputs_answers = outputs[:, input_ids.shape[-1]:]

        # Take only the part of the output that corresponds to the shape of the labels
        outputs_answers = outputs_answers[:, :labels_unpadded.shape[-1]]

        # Take the output that is not ignored  in the labels
        outputs_answers = torch.where(labels_unpadded, outputs_answers, -100)

        # Compute loss
        dummy_loss = torch.tensor(0.0, device=input_ids.device)

        if not return_outputs:
            raise NotImplementedError("Ablations can't compute loss yet!")
        
        # For this https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/trainer.py#L3526
        outputs_answers = torch.cat((torch.zeros_like(outputs_answers[0]).unsqueeze(0), outputs_answers)) 

        return dummy_loss, outputs_answers