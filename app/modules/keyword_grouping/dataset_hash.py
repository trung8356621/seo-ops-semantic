from __future__ import annotations

import hashlib
import json
from typing import Any

from app.core.text.normalization import normalize_text


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def canonical_evidence(evidence: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [
        {"group_type": item["group_type"], "key": item["key"]}
        for item in evidence
    ]
    rows.sort(key=lambda row: (row["key"], row["group_type"]))
    return rows


def canonical_item(
    *,
    ref: str,
    content_hash_value: str,
    language: str | None,
    industry_evidence: list[dict[str, str]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "content_hash": content_hash_value,
        "industry_evidence": canonical_evidence(industry_evidence),
        "language": language,
        "metadata": metadata,
        "ref": ref,
    }


def batch_hash(items: list[dict[str, Any]]) -> str:
    payload = sorted(items, key=lambda row: str(row["ref"]))
    return _sha256({"items": payload})


def dataset_hash(
    scope_ref: str,
    language: str | None,
    items: list[dict[str, Any]],
) -> str:
    payload = {
        "items": sorted(items, key=lambda row: str(row["ref"])),
        "language": language,
        "scope_ref": scope_ref,
    }
    return _sha256(payload)


def _sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
