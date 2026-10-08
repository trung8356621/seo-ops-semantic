"""V3 reciprocal-neighbor grouping fixtures."""

from __future__ import annotations

from app.core.clustering import ClusterPoint
from app.core.text.lexical import pair_evidence
from app.modules.keyword_grouping.hybrid import classify_pair_relation, run_hybrid_semantic_lexical_v1
from tests.unit.test_hybrid_semantic_lexical import _settings, _unit


def _run(texts: dict[str, str], vectors: dict[str, list[float]]):
    points = [ClusterPoint(ref=ref, vector=vectors[ref]) for ref in sorted(texts)]
    return run_hybrid_semantic_lexical_v1(points=points, texts_by_ref=texts, settings=_settings())


def test_factory_near_duplicate_seeds_together() -> None:
    texts = {
        "a": "xưởng may túi xách giá rẻ",
        "b": "xưởng may balo túi xách giá rẻ",
    }
    groups, unassigned, _diag, cfg = _run(
        texts,
        {"a": _unit([1.0, 0.0, 0, 0, 0, 0, 0, 0]), "b": _unit([0.99, 0.01, 0, 0, 0, 0, 0, 0])},
    )
    assert unassigned == []
    assert {m.ref for m in groups[0].members} == {"a", "b"}
    assert cfg["version"] == "v3"
    assert cfg["reciprocal_seed_count"] >= 1


def test_competing_modifiers_are_not_seeds_even_when_reciprocal() -> None:
    texts = {"a": "balo đi học", "b": "balo đi làm"}
    groups, unassigned, diag, _cfg = _run(
        texts,
        {"a": _unit([1.0, 0.0, 0, 0, 0, 0, 0, 0]), "b": _unit([0.99, 0.01, 0, 0, 0, 0, 0, 0])},
    )
    assert groups == []
    assert {u.ref for u in unassigned} == {"a", "b"}
    assert diag.lexical_reject_count >= 1
    relation, _evidence = classify_pair_relation(
        text_a=texts["a"],
        text_b=texts["b"],
        cosine=0.99,
        semantic_floor=0.55,
        containment_min=0.67,
        shared_industry=True,
    )
    assert relation.value == "incompatible"


def test_company_name_and_definition_do_not_join_product_groups() -> None:
    texts = {
        "a": "balo học sinh",
        "b": "balo học sinh tiểu học",
        "c": "Xưởng may balo học sinh tại Hợp Phát",
        "d": "Xưởng may Hợp Phát",
        "e": "balo vải bố",
        "f": "Vải bố là gì",
    }
    school = _unit([1.0, 0.0, 0, 0, 0, 0, 0, 0])
    cloth = _unit([0.0, 1.0, 0, 0, 0, 0, 0, 0])
    groups, unassigned, _diag, _cfg = _run(
        texts,
        {
            "a": school,
            "b": _unit([0.99, 0.01, 0, 0, 0, 0, 0, 0]),
            "c": _unit([0.98, 0.02, 0, 0, 0, 0, 0, 0]),
            "d": school,
            "e": cloth,
            "f": cloth,
        },
    )
    un_refs = {u.ref for u in unassigned}
    assert "d" in un_refs
    assert "f" in un_refs
    for group in groups:
        refs = {m.ref for m in group.members}
        assert "a" not in refs or "d" not in refs
        assert "e" not in refs or "f" not in refs


def test_inventory_size_does_not_change_pair_relation() -> None:
    left, right = "các loại túi xách", "túi xách du lịch"
    small = pair_evidence(left, right)
    site = pair_evidence(left, right, frequent_ngrams=frozenset({"túi xách", "xưởng may", "giá rẻ"}))
    assert small.compatible == site.compatible
    assert small.conflict == site.conflict


def test_dimensions_and_bag_contents_conflict() -> None:
    assert pair_evidence("kích thước balo", "kích thước vali").conflict
    assert pair_evidence("túi đựng mỹ phẩm trong suốt", "túi đựng loa beat box").conflict


def test_reasons_cover_every_unassigned_keyword_once() -> None:
    texts = {
        "a": "balo học sinh",
        "b": "balo học sinh tiểu học",
        "c": "kích thước balo",
        "d": "kích thước vali",
    }
    groups, unassigned, _diag, _cfg = _run(
        texts,
        {
            "a": _unit([1.0, 0, 0, 0, 0, 0, 0, 0]),
            "b": _unit([0.99, 0.01, 0, 0, 0, 0, 0, 0]),
            "c": _unit([0, 1, 0, 0, 0, 0, 0, 0]),
            "d": _unit([0, 0.99, 0.01, 0, 0, 0, 0, 0]),
        },
    )
    assigned = [m.ref for g in groups for m in g.members]
    assert len(assigned) == len(set(assigned))
    assert len(assigned) + len(unassigned) == len(texts)
    assert len({u.ref for u in unassigned}) == len(unassigned)
    assert all(u.reason for u in unassigned)
