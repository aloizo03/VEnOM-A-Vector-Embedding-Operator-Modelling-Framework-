import numpy as np


def selection_k(ratio, n, rounding='ceil'):
    """Number of lake datasets to select for a selection ratio lambda.

    As in the paper (Appendix E), the Vector DB (Qdrant) search selects
    k = ceil(lambda * N), whereas the hand-written similarity functions
    (cosine and Euclidean distance) select k = floor(lambda * N).
    k is always at least 1 and at most N.
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
