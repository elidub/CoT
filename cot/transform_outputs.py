import torch

def transform_outputs(input_tensor: torch.Tensor, A_seq: int, P: int, return_indices = True) -> torch.Tensor:

    B, L = input_tensor.size() # Batch, Length of sentence

    assert P < 0
    assert len(A_seq) == 2

    A0, A1 = A_seq

    # Get the mask where the tokens are A
    mask_A0 = (input_tensor == A0)
    mask_A1 = (input_tensor == A1)

    mask_after_A0 = (mask_A0.cumsum(dim=1) == 1).int()
    mask_after_A1 = (mask_A1.cumsum(dim=1) == 1).int()

    # Cumulative sum on mask, tokens after the first A will have value 1
    mask_after_A = mask_after_A0 + mask_after_A1
    # mask_after_A = (mask_A.cumsum(dim=1) == 1).int()

    # Finding the index of first non-A token after the first A for each row
    indices = torch.argmax(mask_after_A, dim=1)

    # If there are no A_seq token, pad the sentence
    indices_mask = torch.where(indices == 0, L, indices)
    indices_ret = torch.where(indices == 0, -1, indices)

    # Mask the explanation
    mask_fill_P = (torch.arange(input_tensor.size(1)).unsqueeze(0).to(input_tensor.device) > indices_mask.unsqueeze(-1)).expand_as(input_tensor)
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

    return (tokens_padded, indices_ret) if return_indices else tokens_padded

if __name__ == '__main__':
    # Example
    E, A_seq, S, P = 1, (2, 3), 0, -100 # tokens E: explanation, A: answer_seq (e.g. <Answer> <colon>), S: end of sentence, P: padding

    input_tensor = torch.tensor([ # Example input tokens with 
        [E, 10, 11, 12, A_seq[0], A_seq[1], 22,  23,  S, P, P],
        [E, 13, 14, 15, 16, 17,  A_seq[0], A_seq[1], 25, S, P],
        [E, 18, A_seq[0], A_seq[1] , 27, 28, 29,    S,  P, P, P],
        [E, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        [A_seq[0], A_seq[1], 27, 28, 29, 30, 31, 32, 33, 34, 35],
        [P,  P, P,  P,  P,  P,  P,  P,  P,  P,  P],
    ])

    goal_tensor = torch.tensor([
        [   22,   23,    S, P, P, P, P, P, P, P, P],
        [   25,    S,    P, P, P, P, P, P, P, P, P],
        [   27,   28,   29, S, P, P, P, P, P, P, P],
        [   P,     P,    P, P, P, P, P, P, P, P, P],
        [   27, 28, 29, 30, 31, 32, 33, 34, 35, P, P],
        [P,  P, P,  P,  P,  P,  P,  P,  P,  P,  P],
    ])


    tokens_padded, indices = transform_outputs(input_tensor, A_seq, P, return_indices=True)

    print('token\n',   tokens_padded)
    print('indices\n', indices)

    assert (tokens_padded == goal_tensor).all()
