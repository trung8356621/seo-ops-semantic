from __future__ import annotations


def heuristic_confidence(similarity_score: float, assignment_floor: float) -> float:
    """Map cosine similarity to a heuristic confidence in [0, 1].

    This is NOT a probability. It only rescales how far the similarity sits
    above the assignment floor, clipped to [0, 1].
    """
    if similarity_score <= assignment_floor:
        return 0.0
    span = max(1e-9, 1.0 - assignment_floor)
    return max(0.0, min(1.0, (similarity_score - assignment_floor) / span))


def cohesion_score(mean_similarity: float, min_similarity: float) -> float:
    """Simple group cohesion heuristic in [0, 1]. Not a probability."""
    return max(0.0, min(1.0, (0.65 * mean_similarity) + (0.35 * min_similarity)))
