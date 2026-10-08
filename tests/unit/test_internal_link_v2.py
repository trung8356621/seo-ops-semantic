from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.modules.internal_link_v2.contracts import InternalLinkCandidateIn, InternalLinkRankRequest
from app.modules.internal_link_v2.ranker import rank_internal_links


def _candidate(ref: str, **kwargs: object) -> InternalLinkCandidateIn:
    payload = {
        "ref": ref,
        "topic_group_ref": "tg:cotton",
        "url": f"https://example.test/{ref}",
        "eligible": True,
        "inbound_count": 1,
        "relevance": 0.8,
    }
    payload.update(kwargs)
    return InternalLinkCandidateIn.model_validate(payload)


def test_guards_drop_self_ineligible_and_invalid_targets() -> None:
    result = rank_internal_links(
        InternalLinkRankRequest(
            source_ref="article:source",
            candidate_boundary="topic_group",
            candidates=[
                _candidate("article:source", same_as_source=True, relevance=0.99),
                _candidate("article:draft", eligible=False, relevance=0.99),
                _candidate("article:bad-url", url="notaurl", relevance=0.99),
                _candidate("article:dup", already_linked_from_source=True, relevance=0.99),
                _candidate("article:ok", relevance=0.7, inbound_count=1),
            ],
        )
    )
    assert [item.ref for item in result.suggestions] == ["article:ok"]
    assert {item.reason for item in result.rejected} == {
        "self_link",
        "ineligible",
        "invalid_url",
        "duplicate_source_target",
    }


def test_overlinked_target_loses_to_underlinked_relevant_article() -> None:
    result = rank_internal_links(
        InternalLinkRankRequest(
            source_ref="article:source",
            candidate_boundary="topic_group",
            candidates=[
                _candidate("article:heavy", relevance=0.80, inbound_count=40),
                _candidate("article:fresh", relevance=0.78, inbound_count=0),
                _candidate("article:offtopic", relevance=0.45, inbound_count=0),
            ],
        )
    )
    assert [item.ref for item in result.suggestions] == [
        "article:fresh",
        "article:heavy",
        "article:offtopic",
    ]
    assert result.suggestions[0].components.underlinked_bonus > 0
    assert result.suggestions[1].components.overuse_penalty > 0
    assert result.metrics.articles_with_zero_inbound == 2
    assert result.metrics.max_inbound == 40


def test_global_boundary_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InternalLinkRankRequest.model_validate(
            {
                "source_ref": "article:source",
                "candidate_boundary": "global",
                "candidates": [],
            }
        )


def test_repeated_sources_do_not_collapse_onto_one_target() -> None:
    """Comparable relevance rotates the winner. A clear topical gap still beats inbound."""
    profiles = [
        {"fresh": 0.80, "mid": 0.78, "heavy": 0.77},
        {"fresh": 0.74, "mid": 0.81, "heavy": 0.76},
        {"fresh": 0.73, "mid": 0.75, "heavy": 0.90},
        {"fresh": 0.82, "mid": 0.79, "heavy": 0.78},
        {"fresh": 0.70, "mid": 0.84, "heavy": 0.71},
        {"fresh": 0.76, "mid": 0.74, "heavy": 0.75},
    ]
    inbound = {"fresh": 0, "mid": 6, "heavy": 40}
    first: list[str] = []
    for index, scores in enumerate(profiles):
        request = InternalLinkRankRequest(
            source_ref=f"src-{index}",
            candidate_boundary="topic_group",
            limit=1,
            candidates=[
                _candidate(f"article:{name}", relevance=score, inbound_count=inbound[name])
                for name, score in scores.items()
            ],
        )
        once = rank_internal_links(request)
        twice = rank_internal_links(request)
        assert [item.ref for item in once.suggestions] == [item.ref for item in twice.suggestions]
        first.append(once.suggestions[0].ref)

    assert len(set(first)) >= 3
    assert "article:heavy" in first
    assert "article:fresh" in first
    assert first[2] == "article:heavy"
