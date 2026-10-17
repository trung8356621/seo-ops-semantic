from __future__ import annotations

import hashlib
import json
from typing import Any

from app.config import Settings
from app.core.embedding.contracts import EmbeddingProvider
from app.modules.keyword_grouping.analyzer import model_cache_key
from app.modules.keyword_grouping.hybrid import RECIPROCAL_NEIGHBOR_K


def build_fingerprint(
    settings: Settings,
    dataset_hash: str,
    embedding: EmbeddingProvider,
    *,
    scope_ref: str,
    language: str | None,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "algorithm": (settings.keyword_group_algorithm or "").strip().lower(),
        "ambiguity_margin": float(settings.keyword_group_ambiguity_margin),
        "containment_min": float(settings.keyword_group_containment_min),
        "dataset_hash": dataset_hash,
        "language": language,
        "min_group_size": int(settings.keyword_group_min_group_size),
        "model_key": model_cache_key(embedding),
        "reciprocal_neighbor_k": int(RECIPROCAL_NEIGHBOR_K),
        "rescue_semantic_floor": float(settings.keyword_group_rescue_semantic_floor),
        "scope_ref": scope_ref,
        "semantic_floor": float(settings.keyword_group_semantic_floor),
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), payload


def settings_for_fingerprint(settings: Settings, payload: dict[str, Any]) -> Settings:
    return settings.model_copy(update={
        "keyword_group_algorithm": str(payload["algorithm"]),
        "keyword_group_ambiguity_margin": float(payload["ambiguity_margin"]),
        "keyword_group_containment_min": float(payload["containment_min"]),
        "keyword_group_min_group_size": int(payload["min_group_size"]),
        "keyword_group_rescue_semantic_floor": float(payload["rescue_semantic_floor"]),
        "keyword_group_semantic_floor": float(payload["semantic_floor"]),
    })
