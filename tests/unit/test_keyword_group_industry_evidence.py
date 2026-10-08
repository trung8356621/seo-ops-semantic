from __future__ import annotations

import pytest

from app.core.clustering import ClusterPoint
from app.modules.keyword_grouping.analyzer import KeywordGroupAnalyzer, compute_input_hash
from app.modules.keyword_grouping.contracts import (
    IndustryEvidenceIn,
    IndustryMembershipIn,
    KeywordGroupAnalysisRequest,
    KeywordIn,
)
from app.modules.keyword_grouping.hybrid import classify_pair_relation, run_hybrid_semantic_lexical_v1
from tests.unit.test_hybrid_semantic_lexical import ScriptedEmbedding, _settings, _unit


def _membership(ref: str, key: str, group_type: str = "products") -> IndustryEvidenceIn:
    return IndustryEvidenceIn(
        ref=ref,
        memberships=[IndustryMembershipIn(key=key, group_type=group_type)],
    )


def test_request_without_industry_evidence_stays_valid() -> None:
    analyzer = KeywordGroupAnalyzer(
        settings=_settings(TOPIC_EMBEDDING_CACHE_ENABLED=False),
        embedding=ScriptedEmbedding({"balo học sinh": _unit([1, 0, 0, 0, 0, 0, 0, 0])}),
    )
    result = analyzer.analyze(
        KeywordGroupAnalysisRequest(
            scope_ref="4",
            language="vi",
            keywords=[KeywordIn(ref="1", text="balo học sinh")],
        )
    )
    assert result.status == "completed"
    assert result.diagnostics.algorithm_config["industry_evidence"] is False
    assert result.diagnostics.industry_membership_count == 0


def test_unknown_evidence_ref_rejected() -> None:
    with pytest.raises(ValueError, match="not a submitted keyword"):
        KeywordGroupAnalysisRequest(
            scope_ref="4",
            keywords=[KeywordIn(ref="1", text="balo")],
            industry_evidence=[_membership("9", "industry.products.balo")],
        )


def test_duplicate_evidence_ref_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate industry evidence"):
        KeywordGroupAnalysisRequest(
            scope_ref="4",
            keywords=[KeywordIn(ref="1", text="balo")],
            industry_evidence=[
                _membership("1", "industry.products.balo"),
                _membership("1", "industry.products.other"),
            ],
        )


def test_duplicate_membership_key_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate industry membership"):
        KeywordGroupAnalysisRequest(
            scope_ref="4",
            keywords=[KeywordIn(ref="1", text="balo")],
            industry_evidence=[
                IndustryEvidenceIn(
                    ref="1",
                    memberships=[
                        IndustryMembershipIn(key="industry.products.balo", group_type="products"),
                        IndustryMembershipIn(key="industry.products.balo", group_type="products"),
                    ],
                )
            ],
        )


def test_evidence_order_does_not_change_hash_but_membership_change_does() -> None:
    keywords = [
        KeywordIn(ref="2", text="balo laptop"),
        KeywordIn(ref="1", text="balo học sinh"),
    ]
    first = [
        _membership("2", "industry.products.laptop"),
        IndustryEvidenceIn(
            ref="1",
            memberships=[
                IndustryMembershipIn(key="industry.products.school", group_type="products"),
                IndustryMembershipIn(key="industry.products.balo", group_type="products"),
            ],
        ),
    ]
    swapped = [
        IndustryEvidenceIn(
            ref="1",
            memberships=[
                IndustryMembershipIn(key="industry.products.balo", group_type="products"),
                IndustryMembershipIn(key="industry.products.school", group_type="products"),
            ],
        ),
        _membership("2", "industry.products.laptop"),
    ]
    base = compute_input_hash("4", "vi", keywords)
    same = compute_input_hash("4", "vi", list(reversed(keywords)), first)
    other_order = compute_input_hash("4", "vi", keywords, swapped)
    changed = compute_input_hash("4", "vi", keywords, [_membership("1", "industry.products.other")])
    assert same == other_order
    assert same != base
    assert changed != same


def test_shared_broad_industry_group_does_not_override_lexical_conflict() -> None:
    school = "balo học sinh"
    laptop = "balo laptop"
    relation, _evidence = classify_pair_relation(
        text_a=school,
        text_b=laptop,
        cosine=0.99,
        semantic_floor=0.55,
        containment_min=0.67,
        shared_industry=True,
    )
    assert relation.value == "incompatible"

    vector = _unit([1.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    points = [
        ClusterPoint(ref="a", vector=vector),
        ClusterPoint(ref="b", vector=vector),
    ]
    texts = {"a": school, "b": laptop}
    shared = {
        "a": frozenset({"industry.products.balo"}),
        "b": frozenset({"industry.products.balo"}),
    }
    groups, _unassigned, diag, _cfg = run_hybrid_semantic_lexical_v1(
        points=points,
        texts_by_ref=texts,
        settings=_settings(),
        memberships_by_ref=shared,
    )
    for group in groups:
        refs = {member.ref for member in group.members}
        assert refs != {"a", "b"}
    assert diag.industry_supported_edge_count == 0
    assert diag.lexical_reject_count >= 1


def test_shared_industry_group_supports_pair_when_lexical_evidence_is_unknown() -> None:
    left = "widget"
    right = "đặt sản xuất balo cho trường học"
    unknown, _ = classify_pair_relation(
        text_a=left,
        text_b=right,
        cosine=0.9,
        semantic_floor=0.55,
        containment_min=0.67,
        shared_industry=False,
    )
    supported, _ = classify_pair_relation(
        text_a=left,
        text_b=right,
        cosine=0.9,
        semantic_floor=0.55,
        containment_min=0.67,
        shared_industry=True,
    )
    assert unknown.value == "unknown"
    assert supported.value == "compatible"

    vector = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    points = [ClusterPoint(ref="1", vector=vector), ClusterPoint(ref="2", vector=vector)]
    texts = {"1": left, "2": right}
    alone, _u1, diag_alone, _c1 = run_hybrid_semantic_lexical_v1(
        points=points,
        texts_by_ref=texts,
        settings=_settings(),
    )
    grouped, _u2, diag, cfg = run_hybrid_semantic_lexical_v1(
        points=points,
        texts_by_ref=texts,
        settings=_settings(),
        memberships_by_ref={
            "1": frozenset({"industry.services.san_xuat"}),
            "2": frozenset({"industry.services.san_xuat"}),
        },
    )
    assert alone == []
    assert diag_alone.industry_supported_edge_count == 0
    assert [{m.ref for m in g.members} for g in grouped] == [{"1", "2"}]
    assert diag.industry_evidence_keyword_count == 2
    assert diag.industry_membership_count == 2
    assert diag.industry_supported_edge_count == 1
    assert cfg["industry_evidence"] is True


def test_below_semantic_floor_industry_share_stays_unknown() -> None:
    relation, _ = classify_pair_relation(
        text_a="widget",
        text_b="đặt sản xuất balo cho trường học",
        cosine=0.2,
        semantic_floor=0.55,
        containment_min=0.67,
        shared_industry=True,
    )
    assert relation.value == "unknown"
