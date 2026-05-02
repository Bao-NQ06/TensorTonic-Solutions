import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here

    if len(seqs) == 0:
        return np.array([])

    # Determine max length
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)

    N = len(seqs)
    L = max_len

    # Initialize output array with pad_value
    result = np.full((N, L), pad_value)

    # Fill with actual sequence values
    for i, seq in enumerate(seqs):
        trunc = seq[:L]  # truncate if longer than max_len
        result[i, :len(trunc)] = trunc

    return result