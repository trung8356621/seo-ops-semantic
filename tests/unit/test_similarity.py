from __future__ import annotations

import pytest

from app.core.similarity import cosine_similarity


def test_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])
