import torch
from typing import List


def greedy_ctc_decode(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_idx: int = 0
) -> List[List[int]]:
    """
    Greedy CTC decoding algorithm.

    Performs CTC decoding by:
    1. Taking argmax at each timestep
    2. Merging repeated characters
    3. Removing blank tokens

    Args:
        logits: Log probabilities from model (batch, time, vocab_size)
        input_lengths: Valid sequence lengths for each batch item (batch,)
        blank_idx: Index of the CTC blank token (default 0)

    Returns:
        List of decoded integer sequences, one per batch item
    """
    # Get the most likely character at each timestep
    preds = torch.argmax(logits, dim=-1)  # (batch, time)
    batch_size = logits.shape[0]

    results = []

    for i in range(batch_size):
        # Extract predictions up to valid sequence length
        pred = preds[i, :input_lengths[i]]

        # Step 1: Merge repeated characters (CTC collapse rule)
        merged = []
        prev_token = -1  # Initialize with impossible token value

        for token in pred:
            if token != prev_token:
                merged.append(token.item())
            prev_token = token

        # Step 2: Remove blank tokens
        decoded = [token for token in merged if token != blank_idx]

        results.append(decoded)

    return results
