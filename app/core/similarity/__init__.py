from __future__ import annotations

import math
from typing import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) == 0 or len(b) == 0:
        raise ValueError("vectors must be non-empty")
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimensions")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for left, right in zip(a, b, strict=True):
        left_f = float(left)
        right_f = float(right)
        if not math.isfinite(left_f) or not math.isfinite(right_f):
            raise ValueError("vectors must contain finite numbers")
        dot += left_f * right_f
        norm_a += left_f * left_f
        norm_b += right_f * right_f

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
