from __future__ import annotations

from app.modules.keyword_grouping.analyzer import content_hash as analyzer_content_hash
from app.modules.keyword_grouping.dataset_hash import content_hash, dataset_hash, canonical_item


def test_content_hash_matches_embedding_cache_hash() -> None:
    assert content_hash("  Máy   phát ") == analyzer_content_hash("  Máy   phát ")


def test_dataset_hash_changes_when_ref_set_changes() -> None:
    language = "vi"
    first = dataset_hash("6", language, [_item("1", "máy phát"), _item("2", "ổn áp")])
    second = dataset_hash("6", language, [_item("1", "máy phát")])
    assert first != second


def test_dataset_hash_ignores_item_order() -> None:
    items = [_item("2", "ổn áp"), _item("1", "máy phát")]
    reversed_items = list(reversed(items))
    assert dataset_hash("6", "vi", items) == dataset_hash("6", "vi", reversed_items)


def _item(ref: str, text: str) -> dict:
    return canonical_item(
        ref=ref,
        content_hash_value=content_hash(text),
        language="vi",
        industry_evidence=[],
        metadata={},
    )
