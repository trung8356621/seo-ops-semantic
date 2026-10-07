"""Calibration-oriented checks: lexical vs semantic evidence stay distinct."""

from __future__ import annotations

from typing import Sequence

import pytest

from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text
from app.modules.concept_matching.analyzer import ConceptMatchAnalyzer
from app.modules.concept_matching.contracts import ConceptMatchAnalysisRequest


class MappingEmbeddingProvider:
    PROVIDER_KEY = "fake"
    MODEL_KEY = "fake-calib"
    MODEL_VERSION = "v0"
    DIMENSIONS = 2

    def __init__(self, mapping: dict[str, tuple[float, ...]]) -> None:
        self._mapping = {normalize_text(k): v for k, v in mapping.items()}
        self.batch_calls = 0

    @property
    def provider_key(self) -> str:
        return self.PROVIDER_KEY

    @property
    def model_key(self) -> str:
        return self.MODEL_KEY

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION

    @property
    def dimensions(self) -> int | None:
        return self.DIMENSIONS

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        self.batch_calls += 1
        out: list[EmbeddingResult] = []
        for raw in texts:
            key = normalize_text(str(raw))
            out.append(
                EmbeddingResult(
                    vector=self._mapping[key],
                    model_key=self.MODEL_KEY,
                    model_version=self.MODEL_VERSION,
                    dimensions=self.DIMENSIONS,
                    provider=self.PROVIDER_KEY,
                )
            )
        return out


def test_service_semantic_directional_evidence() -> None:
    """Service intent should score closer to manufacturing positives than shop noise."""
    provider = MappingEmbeddingProvider(
        {
            "xưởng may balo theo yêu cầu": (1.0, 0.0),
            "balo đẹp giá rẻ": (0.0, 1.0),
            "xưởng may balo": (0.98, 0.1),
            "may balo theo yêu cầu": (0.95, 0.1),
            "gia công balo": (0.9, 0.15),
            "đặt may số lượng lớn": (0.88, 0.2),
            "balo đẹp": (0.1, 0.95),
            "giá balo": (0.05, 0.9),
            "mua balo": (0.0, 0.92),
        }
    )
    response = ConceptMatchAnalyzer(provider).analyze(
        ConceptMatchAnalysisRequest.model_validate(
            {
                "scope_ref": "calib:service",
                "entities": [
                    {"ref": "svc", "text": "xưởng may balo theo yêu cầu"},
                    {"ref": "noise", "text": "balo đẹp giá rẻ"},
                ],
                "concepts": [
                    {
                        "key": "industry.services.manufacturing.test",
                        "match_mode": "semantic",
                        "matching_strategy": "semantic",
                        "positive_examples": [
                            "xưởng may balo",
                            "may balo theo yêu cầu",
                            "gia công balo",
                            "đặt may số lượng lớn",
                        ],
                        "negative_examples": ["balo đẹp", "giá balo", "mua balo"],
                    }
                ],
            }
        )
    )
    svc = response.entities[0].concepts[0]
    noise = response.entities[1].concepts[0]
    assert svc.lexical.matched is False  # match_mode=semantic
    assert svc.positive_max is not None and noise.positive_max is not None
    assert svc.positive_max > noise.positive_max
    assert svc.margin is not None and noise.margin is not None
    assert svc.margin > noise.margin


def test_cta_semantic_evidence_separated_from_lexical() -> None:
    provider = MappingEmbeddingProvider(
        {
            "liên hệ ngay": (1.0, 0.0),
            "nhận tư vấn ngay": (0.95, 0.05),
            "đăng ký nhận báo giá": (0.7, 0.3),
            "báo giá balo theo yêu cầu": (0.2, 0.8),
            "giá may balo số lượng lớn": (0.15, 0.85),
            "mua balo học sinh": (0.1, 0.9),
            "nhận tư vấn": (0.97, 0.03),
            "đăng ký ngay": (0.9, 0.1),
            "click vào đây": (0.85, 0.1),
            "gọi ngay": (0.88, 0.1),
            "báo giá sản phẩm": (0.25, 0.75),
            "giá sản phẩm": (0.2, 0.8),
            "mua sản phẩm": (0.15, 0.85),
            "xưởng sản xuất": (0.1, 0.7),
        }
    )
    entities = [
        "liên hệ ngay",
        "nhận tư vấn ngay",
        "đăng ký nhận báo giá",
        "báo giá balo theo yêu cầu",
        "giá may balo số lượng lớn",
        "mua balo học sinh",
    ]
    response = ConceptMatchAnalyzer(provider).analyze(
        ConceptMatchAnalysisRequest.model_validate(
            {
                "scope_ref": "calib:cta",
                "entities": [{"ref": f"e{i}", "text": t} for i, t in enumerate(entities)],
                "concepts": [
                    {
                        "key": "system.cta.test",
                        "match_mode": "semantic",
                        "matching_strategy": "semantic",
                        "positive_examples": [
                            "liên hệ ngay",
                            "nhận tư vấn",
                            "đăng ký ngay",
                            "click vào đây",
                            "gọi ngay",
                        ],
                        "negative_examples": [
                            "báo giá sản phẩm",
                            "giá sản phẩm",
                            "mua sản phẩm",
                            "xưởng sản xuất",
                        ],
                    }
                ],
            }
        )
    )
    scores = [e.concepts[0] for e in response.entities]
    for score in scores:
        assert "matched" in score.lexical.model_dump()
        assert score.positive_max is not None
    # CTA-like entities should outrank shop/price noise on positive_max (directional).
    cta_max = max(scores[i].positive_max for i in range(3))
    noise_max = max(scores[i].positive_max for i in range(3, 6))
    assert cta_max > noise_max
