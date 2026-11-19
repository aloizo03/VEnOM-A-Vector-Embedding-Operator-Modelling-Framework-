import numpy as np


def selection_k(ratio, n, rounding='ceil'):
    """Number of lake datasets to select for a selection ratio lambda.

    Single definition used by every selection path (Qdrant, Euclidean, cosine,
    K-Means bounds, random/SR) so all methods select the same k:
        k = max(1, ceil(lambda * N)), capped at N.
    `ratio` may be given as a fraction (0.2) or a percentage (20).
    """
    if ratio > 1:
        ratio = ratio / 100
    if n <= 0:
        return 0
    if rounding == 'floor':
        k = int(np.floor(ratio * n))
    else:
        k = int(np.ceil(ratio * n))
    return min(n, max(1, k))
