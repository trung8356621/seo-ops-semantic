from __future__ import annotations

import pytest

from app.modules.concept_matching.contracts import ConceptDecisionPolicy
from app.modules.concept_matching.scoring import score_entity_against_concept


def test_positive_max_and_top_k_mean() -> None:
    evidence = score_entity_against_concept(
        entity_vector=[1.0, 0.0, 0.0],
        positive_examples=["a", "b", "c", "d"],
        positive_vectors=[
            [1.0, 0.0, 0.0],  # 1.0
            [0.8, 0.6, 0.0],  # 0.8
            [0.6, 0.8, 0.0],  # 0.6
            [0.0, 1.0, 0.0],  # 0.0
        ],
        negative_examples=[],
        negative_vectors=[],
    )
    assert evidence.positive_max == pytest.approx(1.0)
    assert evidence.best_positive_example == "a"
    assert evidence.best_positive_similarity == pytest.approx(1.0)
    # top-3 of [1.0, 0.8, 0.6, 0.0] => mean 0.8
    assert evidence.positive_top_k_mean == pytest.approx(0.8)
    assert evidence.negative_max is None
    assert evidence.margin is None
    assert evidence.suggested_match is None


def test_negative_score_and_margin() -> None:
    evidence = score_entity_against_concept(
        entity_vector=[1.0, 0.0],
        positive_examples=["pos"],
        positive_vectors=[[1.0, 0.0]],
        negative_examples=["neg"],
        negative_vectors=[[0.0, 1.0]],
    )
    assert evidence.positive_max == pytest.approx(1.0)
    assert evidence.negative_max == pytest.approx(0.0)
    assert evidence.margin == pytest.approx(1.0)
    assert evidence.best_negative_example == "neg"


def test_decision_policy_with_and_without_negatives() -> None:
    policy = ConceptDecisionPolicy(min_positive_score=0.7, min_margin=0.05)

    match = score_entity_against_concept(
        entity_vector=[1.0, 0.0],
        positive_examples=["pos"],
        positive_vectors=[[1.0, 0.0]],
        negative_examples=["neg"],
        negative_vectors=[[0.0, 1.0]],  # orthogonal → sim 0.0
        decision_policy=policy,
    )
    assert match.margin == pytest.approx(1.0)
    assert match.suggested_match is True

    no_margin = score_entity_against_concept(
        entity_vector=[1.0, 0.0],
        positive_examples=["pos"],
        positive_vectors=[[0.8, 0.6]],  # sim 0.8
        negative_examples=["neg"],
        negative_vectors=[[0.78, 0.6255]],  # ~0.78 → margin ~0.02 < 0.05
        decision_policy=policy,
    )
    assert no_margin.positive_max == pytest.approx(0.8)
    assert no_margin.margin is not None and no_margin.margin < 0.05
    assert no_margin.suggested_match is False

    no_neg = score_entity_against_concept(
        entity_vector=[1.0, 0.0],
        positive_examples=["pos"],
        positive_vectors=[[0.8, 0.0]],
        negative_examples=[],
        negative_vectors=[],
        decision_policy=policy,
    )
    assert no_neg.suggested_match is True
