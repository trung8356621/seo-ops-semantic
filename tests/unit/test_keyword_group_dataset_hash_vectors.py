from __future__ import annotations

import json
from pathlib import Path

from app.modules.keyword_grouping.dataset_hash import canonical_item, content_hash, dataset_hash


def test_php_golden_vectors_match_python() -> None:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "keyword_group_dataset_hash_vectors.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert len(cases) == 6
    for case in cases:
        items = [
            canonical_item(
                ref=item["ref"],
                content_hash_value=content_hash(item["text"]),
                language=item["language"],
                industry_evidence=item["industry_evidence"],
                metadata=item["metadata"],
            )
            for item in case["items"]
        ]
        assert dataset_hash(case["scope_ref"], case["language"], items) == case["dataset_hash"]
        for item, canonical in zip(case["items"], items, strict=True):
            assert item["content_hash"] == canonical["content_hash"]
