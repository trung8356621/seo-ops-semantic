from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.modules.topic.contracts import TopicAnalysisRequest, TopicKeywordIn


def test_vietnamese_input_accepted() -> None:
    req = TopicAnalysisRequest(
        site_ref="6",
        language="vi",
        keywords=[TopicKeywordIn(ref="kw-1", text="balo học sinh")],
    )
    assert req.keywords[0].text == "balo học sinh"


def test_duplicate_refs_rejected() -> None:
    with pytest.raises(ValidationError):
        TopicAnalysisRequest(
            site_ref="6",
            keywords=[
                TopicKeywordIn(ref="kw-1", text="a"),
                TopicKeywordIn(ref="kw-1", text="b"),
            ],
        )


def test_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        TopicKeywordIn(ref="kw-1", text="   ")


def test_refs_preserved() -> None:
    req = TopicAnalysisRequest(
        site_ref="site-a",
        keywords=[
            TopicKeywordIn(ref="kw-10", text="balo quà tặng"),
            TopicKeywordIn(ref="kw-2", text="túi xách"),
        ],
    )
    assert [k.ref for k in req.keywords] == ["kw-10", "kw-2"]
