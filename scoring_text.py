import numpy as np
from collections import Counter

def compute_length_score(x_i):
    """
    Difficulty score based on the real (non-padded) length of the sequence.
    Longer sequences are considered more difficult.
    """
    return np.count_nonzero(x_i)


def compute_infrequency_score(x_i, token_freq):
    """
    Difficulty score based on inverse word frequency.
    Sequences with rarer tokens are considered more difficult.
    """
    return sum(1.0 / (token_freq.get(token, 1e-6)) for token in x_i if token != 0)


def get_token_frequency(x_data):
    """
    Computes frequency of each token in the dataset.
    """
    token_counts = Counter(int(i) for row in x_data for i in row if i != 0)
    total_tokens = sum(token_counts.values())
    return {token: count / total_tokens for token, count in token_counts.items()}


def get_combined_scores(x_data, alpha=0.5):
    """
    Returns a combined difficulty score for each sequence in x_data.
    The score is a weighted combination of length and inverse frequency.

    Parameters:
    - x_data: list of tokenized and padded sequences
    - alpha: weighting factor for length score (0 ≤ alpha ≤ 1)

    Returns:
    - combined_scores: np.array of difficulty scores
    """
    length_scores = np.array([compute_length_score(x_i) for x_i in x_data])
    
    token_freq = get_token_frequency(x_data)
    infrequency_scores = np.array([
        compute_infrequency_score(x_i, token_freq) for x_i in x_data
    ])

    # Normalize scores to [0, 1]
    length_scores_norm = (length_scores - np.min(length_scores)) / (np.ptp(length_scores) + 1e-6)
    infrequency_scores_norm = (infrequency_scores - np.min(infrequency_scores)) / (np.ptp(infrequency_scores) + 1e-6)

    print("Length scores (normalized):", length_scores_norm)
    # Weighted combination
    combined_scores = alpha * length_scores_norm + (1 - alpha) * infrequency_scores_norm
    return combined_scores