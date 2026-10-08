"""Grouping precision: competing modifiers must not share a group."""

from __future__ import annotations

from app.core.clustering import ClusterPoint
from app.core.text.lexical import pair_evidence
from app.modules.keyword_grouping.hybrid import classify_pair_relation, run_hybrid_semantic_lexical_v1
from tests.unit.test_hybrid_semantic_lexical import _settings, _unit


def _pair(a: str, b: str, cosine: float = 0.9) -> str:
    relation, _evidence = classify_pair_relation(
        text_a=a,
        text_b=b,
        cosine=cosine,
        semantic_floor=0.55,
        containment_min=0.67,
        shared_industry=True,
    )
    return relation.value


def test_competing_modifiers_conflict_even_with_shared_ngram_and_industry() -> None:
    assert pair_evidence("balo đi học", "balo đi làm").conflict
    assert pair_evidence("kích thước balo", "kích thước vali").conflict
    assert pair_evidence("túi đựng mỹ phẩm trong suốt", "túi đựng loa beat box").conflict
    assert _pair("balo đi học", "balo đi làm") == "incompatible"
    assert _pair("kích thước balo", "kích thước vali") == "incompatible"
    assert _pair("túi đựng mỹ phẩm trong suốt", "túi đựng loa beat box") == "incompatible"


def test_school_backpack_variations_stay_together() -> None:
    texts = {
        "a": "balo học sinh",
        "b": "balo học sinh tiểu học",
        "c": "balo cho học sinh cấp 1",
    }
    points = [
        ClusterPoint(ref="a", vector=_unit([1.0, 0.0, 0, 0, 0, 0, 0, 0])),
        ClusterPoint(ref="b", vector=_unit([0.99, 0.01, 0, 0, 0, 0, 0, 0])),
        ClusterPoint(ref="c", vector=_unit([0.98, 0.02, 0, 0, 0, 0, 0, 0])),
    ]
    groups, unassigned, _diag, _cfg = run_hybrid_semantic_lexical_v1(
        points=points,
        texts_by_ref=texts,
        settings=_settings(),
    )
    assert unassigned == []
    assert len(groups) == 1
    assert {m.ref for m in groups[0].members} == {"a", "b", "c"}


def test_broad_shared_anchor_does_not_chain_unrelated_phrases() -> None:
    frequent = frozenset({"túi xách"})
    left = pair_evidence("các loại túi xách", "túi xách du lịch", frequent_ngrams=frequent)
    right = pair_evidence("túi xách du lịch", "túi xách quảng cáo", frequent_ngrams=frequent)
    assert left.conflict and right.conflict
    relation, _evidence = classify_pair_relation(
        text_a="túi xách du lịch",
        text_b="túi xách quảng cáo",
        cosine=0.9,
        semantic_floor=0.55,
        containment_min=0.67,
        shared_industry=True,
        frequent_ngrams=frequent,
    )
    assert relation.value == "incompatible"


def test_shuffled_input_is_deterministic() -> None:
    texts = {
        "a": "balo học sinh",
        "b": "balo học sinh tiểu học",
        "c": "vali kéo du lịch",
        "d": "vali kéo",
    }
    vectors = {
        "a": _unit([1.0, 0.0, 0, 0, 0, 0, 0, 0]),
        "b": _unit([0.98, 0.02, 0, 0, 0, 0, 0, 0]),
        "c": _unit([0.0, 1.0, 0, 0, 0, 0, 0, 0]),
        "d": _unit([0.02, 0.98, 0, 0, 0, 0, 0, 0]),
    }
    order_a = ["a", "b", "c", "d"]
    order_b = ["d", "b", "a", "c"]

    def run(order: list[str]):
        points = [ClusterPoint(ref=ref, vector=vectors[ref]) for ref in order]
        groups, unassigned, _diag, _cfg = run_hybrid_semantic_lexical_v1(
            points=points,
            texts_by_ref=texts,
            settings=_settings(),
        )
        return (
            [frozenset(m.ref for m in group.members) for group in groups],
            [u.ref for u in unassigned],
        )

    assert run(order_a) == run(order_b)
