import torch

E, A, S, P = 1, 2, 0, -100 # tokens E: explanation, A: answer, S: end of sentence, P: padding

input_tensor = torch.tensor([ # Example input tokens with 
    [E, 10, 11, 12, A,  21, 22,  23,  S, P, P],
    [E, 13, 14, 15, 16, 17, A,   24, 25, S, P],
    [E, 18, A,  26, 27, 28, S,    P,  P, P, P]
])

goal_tensor = torch.tensor([
    [  21,   22,   23,    S, P, P, P, P, P, P, P],
    [  24,   25,    S,    P, P, P, P, P, P, P, P],
    [  26,   27,   28,    S, P, P, P, P, P, P, P]
])


def process_tensor(input_tensor: torch.Tensor, E: int, A: int, S: int, P: int) -> torch.Tensor:

    assert P < 0

    # Get the mask where the tokens are A
    mask_A = (input_tensor == A)

     # Cumulative sum on mask, tokens after the first A will have value 1
    mask_after_A = (mask_A.cumsum(dim=1) == 1).int()

    # Finding the index of first non-A token after the first A for each row
    indices = torch.argmax(mask_after_A, dim=1).unsqueeze(-1)

    # Not sure if this is necessary, but it will make sure that the indices are not out of bounds ?
    # index_tensor = indices + torch.arange(input_tensor.size(1)).unsqueeze(0).to(input_tensor.device)

    # Mask the explanation
    mask_fill_P = (torch.arange(input_tensor.size(1)).unsqueeze(0).to(input_tensor.device) > indices).expand_as(input_tensor)
    output_tensor = torch.where(mask_fill_P, input_tensor, P)

    # Where the answer 1, assuming thaat all tokens are >= 0
    positive_mask = output_tensor >= 0

    # Lengths of the answers
    lengths = torch.sum(positive_mask, dim = 1)

    # All answers in a flat tensor
    tokens_flat = output_tensor[positive_mask]

    # Split the flat tensor based on lengths and padd them with P
    y = torch.split(tokens_flat, lengths.tolist(), dim = 0)
    tokens_block = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=P)

    # Pad the block with P to original size
    tokens_padded = torch.nn.functional.pad(tokens_block, (0, output_tensor.shape[1] - tokens_block.shape[1]), mode='constant', value=P)

    return tokens_padded



tokens_padded = process_tensor(input_tensor, E, A, S, P)

assert (tokens_padded == goal_tensor).all()
