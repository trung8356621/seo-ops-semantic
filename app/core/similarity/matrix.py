from __future__ import annotations

from typing import Sequence

import numpy as np


def pairwise_cosine_similarity(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Return an (n, n) cosine similarity matrix for finite non-zero vectors."""
    if not vectors:
        return np.zeros((0, 0), dtype=np.float64)
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero vectors are not allowed")
    normalized = matrix / norms
    return normalized @ normalized.T
