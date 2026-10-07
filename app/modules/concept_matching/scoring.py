from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.core.similarity.cosine import cosine_similarity
from app.modules.concept_matching.contracts import ConceptDecisionPolicy


POSITIVE_TOP_K = 3


@dataclass(frozen=True, slots=True)
class ConceptScoreEvidence:
    positive_max: float
    positive_top_k_mean: float
    negative_max: float | None
    margin: float | None
    best_positive_example: str
    best_positive_similarity: float
    best_negative_example: str | None
    best_negative_similarity: float | None
    suggested_match: bool | None


def score_entity_against_concept(
    *,
    entity_vector: Sequence[float],
    positive_examples: Sequence[str],
    positive_vectors: Sequence[Sequence[float]],
    negative_examples: Sequence[str],
    negative_vectors: Sequence[Sequence[float]],
    decision_policy: ConceptDecisionPolicy | None = None,
) -> ConceptScoreEvidence:
    if len(positive_examples) == 0:
        raise ValueError("positive_examples must not be empty")
    if len(positive_examples) != len(positive_vectors):
        raise ValueError("positive_examples and positive_vectors length mismatch")
    if len(negative_examples) != len(negative_vectors):
        raise ValueError("negative_examples and negative_vectors length mismatch")

    positive_sims = [
        cosine_similarity(entity_vector, vector) for vector in positive_vectors
    ]
    best_pos_idx = max(range(len(positive_sims)), key=lambda i: positive_sims[i])
    positive_max = float(positive_sims[best_pos_idx])
    k = min(POSITIVE_TOP_K, len(positive_sims))
    top_k = sorted(positive_sims, reverse=True)[:k]
    positive_top_k_mean = float(sum(top_k) / k)

    negative_max: float | None = None
    margin: float | None = None
    best_negative_example: str | None = None
    best_negative_similarity: float | None = None

    if negative_examples:
        negative_sims = [
            cosine_similarity(entity_vector, vector) for vector in negative_vectors
        ]
        best_neg_idx = max(range(len(negative_sims)), key=lambda i: negative_sims[i])
        negative_max = float(negative_sims[best_neg_idx])
        best_negative_example = negative_examples[best_neg_idx]
        best_negative_similarity = negative_max
        margin = float(positive_max - negative_max)

    suggested: bool | None = None
    if decision_policy is not None:
        # Kept for unit tests / direct callers. Analyzer uses decision.apply_semantic_gate.
        if decision_policy.min_positive_score is None:
            suggested = False
        else:
            suggested = positive_max >= decision_policy.min_positive_score
            if suggested and negative_max is not None and margin is not None:
                suggested = margin >= decision_policy.min_margin

    return ConceptScoreEvidence(
        positive_max=positive_max,
        positive_top_k_mean=positive_top_k_mean,
        negative_max=negative_max,
        margin=margin,
        best_positive_example=positive_examples[best_pos_idx],
        best_positive_similarity=positive_max,
        best_negative_example=best_negative_example,
        best_negative_similarity=best_negative_similarity,
        suggested_match=suggested,
    )
