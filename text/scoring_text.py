import numpy as np
from collections import Counter

def compute_length_score(x_i):
    return np.count_nonzero(x_i)


def compute_infrequency_score(x_i, token_freq):
    return sum(1.0 / (token_freq.get(token, 1e-6)) for token in x_i if token != 0)


def get_token_frequency(x_data):
    token_counts = Counter(int(i) for row in x_data for i in row if i != 0)
    total_tokens = sum(token_counts.values())
    return {token: count / total_tokens for token, count in token_counts.items()}


def get_length_scores(x_data):
    length_scores = np.array([compute_length_score(x_i) for x_i in x_data])
    length_scores_norm = (length_scores - np.min(length_scores)) / (np.ptp(length_scores) + 1e-6)
    return length_scores_norm


def get_infrequency_scores(x_data):
    token_freq = get_token_frequency(x_data)
    infrequency_scores = np.array([
        compute_infrequency_score(x_i, token_freq) for x_i in x_data
    ])
    infrequency_scores_norm = (infrequency_scores - np.min(infrequency_scores)) / (np.ptp(infrequency_scores) + 1e-6)
    return infrequency_scores_norm


def get_combined_scores(x_data, alpha=0.5):
    length_scores_norm = get_length_scores(x_data)
    infrequency_scores_norm = get_infrequency_scores(x_data)
    
    combined_scores = alpha * length_scores_norm + (1 - alpha) * infrequency_scores_norm
    return combined_scores
