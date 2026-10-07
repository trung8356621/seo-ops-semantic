from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.modules.concept_matching.contracts import ConceptMatchAnalysisRequest


def _base(**overrides: object) -> dict:
    payload: dict = {
        "scope_ref": "site:4",
        "language": "vi",
        "entities": [{"ref": "kw_1", "text": "tuyển nhân viên may balo"}],
        "concepts": [
            {
                "key": "custom.recruitment",
                "positive_examples": ["tuyển thợ may", "việc làm xưởng may"],
                "negative_examples": ["xưởng may balo"],
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_valid_request() -> None:
    req = ConceptMatchAnalysisRequest.model_validate(_base())
    assert req.entities[0].ref == "kw_1"
    assert len(req.concepts[0].positive_examples) == 2


def test_duplicate_entity_refs_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        ConceptMatchAnalysisRequest.model_validate(
            _base(
                entities=[
                    {"ref": "kw_1", "text": "a"},
                    {"ref": "kw_1", "text": "b"},
                ]
            )
        )
    assert "duplicate entity refs" in str(exc.value)


def test_duplicate_concept_keys_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        ConceptMatchAnalysisRequest.model_validate(
            _base(
                concepts=[
                    {"key": "custom.a", "positive_examples": ["x"]},
                    {"key": "custom.a", "positive_examples": ["y"]},
                ]
            )
        )
    assert "duplicate concept keys" in str(exc.value)


def test_positive_examples_required() -> None:
    with pytest.raises(ValidationError):
        ConceptMatchAnalysisRequest.model_validate(
            _base(concepts=[{"key": "custom.a", "positive_examples": []}])
        )
    with pytest.raises(ValidationError):
        ConceptMatchAnalysisRequest.model_validate(
            _base(concepts=[{"key": "custom.a", "positive_examples": ["  ", ""]}])
        )


def test_normalization_and_dedupe_examples() -> None:
    req = ConceptMatchAnalysisRequest.model_validate(
        _base(
            concepts=[
                {
                    "key": "custom.a",
                    "positive_examples": ["  tuyển thợ may  ", "tuyển thợ may", "việc làm"],
                    "negative_examples": ["xưởng", "  xưởng  ", ""],
                }
            ]
        )
    )
    assert req.concepts[0].positive_examples == ["tuyển thợ may", "việc làm"]
    assert req.concepts[0].negative_examples == ["xưởng"]


def test_empty_entity_text_rejected() -> None:
    with pytest.raises(ValidationError):
        ConceptMatchAnalysisRequest.model_validate(
            _base(entities=[{"ref": "kw_1", "text": "   "}])
        )
