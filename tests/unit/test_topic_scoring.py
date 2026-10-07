from __future__ import annotations

from app.modules.topic.scoring import cohesion_score, heuristic_confidence


def test_confidence_is_not_raw_similarity() -> None:
    assert heuristic_confidence(0.87, 0.62) != 0.87
    assert 0.0 <= heuristic_confidence(0.87, 0.62) <= 1.0
    assert heuristic_confidence(0.50, 0.62) == 0.0


def test_cohesion_bounded() -> None:
    value = cohesion_score(0.9, 0.8)
    assert 0.0 <= value <= 1.0
