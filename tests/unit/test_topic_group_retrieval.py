from __future__ import annotations

from typing import Sequence

import pytest

from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text
from app.modules.topic_group_retrieval.contracts import (
    TopicGroupCandidateIn,
    TopicGroupMatchPolicyIn,
    TopicGroupMatchRequest,
)
from app.modules.topic_group_retrieval.matcher import TopicGroupMatcher, segment_content


class MappingEmbeddingProvider:
    def __init__(self, mapping: dict[str, tuple[float, ...]], default: tuple[float, ...] = (0.0, 1.0)) -> None:
        self._mapping = {normalize_text(key): value for key, value in mapping.items()}
        self._default = default

    @property
    def provider_key(self) -> str:
        return "fake"

    @property
    def model_key(self) -> str:
        return "fake-group"

    @property
    def model_version(self) -> str:
        return "v0"

    @property
    def dimensions(self) -> int | None:
        return len(self._default)

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        out: list[EmbeddingResult] = []
        for raw in texts:
            vector = self._mapping.get(normalize_text(raw), self._default)
            out.append(
                EmbeddingResult(
                    vector=vector,
                    model_key="fake-group",
                    model_version="v0",
                    dimensions=len(self._default),
                    provider="fake",
                )
            )
        return out


def test_topic_group_ranking_preserves_external_refs() -> None:
    matcher = TopicGroupMatcher(
        MappingEmbeddingProvider(
            {
                "áo thun cotton": (1.0, 0.0),
                "cotton t-shirt": (0.95, 0.05),
                "áo thun cotton nam": (0.9, 0.1),
                "máy lọc nước": (0.0, 1.0),
            }
        )
    )
    result = matcher.match(
        TopicGroupMatchRequest(
            scope_ref="site:ext-1",
            query="áo thun cotton nam",
            groups=[
                TopicGroupCandidateIn(ref="tg:cotton", label="Cotton", examples=["áo thun cotton", "cotton t-shirt"]),
                TopicGroupCandidateIn(ref="tg:water", label="Water", examples=["máy lọc nước"]),
            ],
            policy=TopicGroupMatchPolicyIn(min_score=0.55, limit=2),
        )
    )
    assert len(result.matches) > 0
    assert result.matches[0].ref == "tg:cotton"
    assert result.scope_ref == "site:ext-1"
    assert all(item.ref.startswith("tg:") for item in result.matches)
    assert "tg:water" not in [item.ref for item in result.matches]


def test_case_b_direct_query_exact_name_match() -> None:
    """Case B: Direct query 'balo học sinh' gives exact group name deterministic priority."""
    matcher = TopicGroupMatcher(
        MappingEmbeddingProvider(
            {
                "balo học sinh": (1.0, 0.0),
                "balo đi học": (0.8, 0.2),
                "chất liệu vải": (0.1, 0.9),
            }
        )
    )
    result = matcher.match(
        TopicGroupMatchRequest(
            scope_ref="site:4",
            query="balo học sinh",
            groups=[
                TopicGroupCandidateIn(ref="g:school", label="balo học sinh", examples=["balo học sinh", "cặp học sinh"]),
                TopicGroupCandidateIn(ref="g:general", label="balo đi học", examples=["balo đi học"]),
                TopicGroupCandidateIn(ref="g:fabric", label="chất liệu vải", examples=["chất liệu vải"]),
            ],
            policy=TopicGroupMatchPolicyIn(min_score=0.55, limit=3),
        )
    )
    assert len(result.matches) >= 1
    top = result.matches[0]
    assert top.ref == "g:school"
    assert top.score >= 0.95
    assert top.evidence.exact_name_matched is True
    assert top.evidence.lexical_matched is True
    # 'chất liệu vải' must be excluded below min_score
    refs = [m.ref for m in result.matches]
    assert "g:fabric" not in refs


def test_case_c_lexical_distractor_does_not_dominate() -> None:
    """Case C: Generic phrase overlap with weak semantic relevance cannot dominate."""
    matcher = TopicGroupMatcher(
        MappingEmbeddingProvider(
            {
                # Query: article discussing manufacturing school backpacks
                "xưởng may balo học sinh cấp 2": (0.9, 0.1),
                # Group A: true semantic match
                "balo học sinh": (0.95, 0.05),
                # Group B: distractor with incidental word 'may' or 'xưởng'
                "xưởng": (0.2, 0.8),
                "dụng cụ cơ khí": (0.1, 0.9),
            }
        )
    )
    result = matcher.match(
        TopicGroupMatchRequest(
            scope_ref="site:4",
            query="xưởng may balo học sinh cấp 2",
            groups=[
                TopicGroupCandidateIn(ref="g:school", label="balo học sinh", examples=["balo học sinh"]),
                TopicGroupCandidateIn(ref="g:distractor", label="dụng cụ cơ khí", examples=["dụng cụ cơ khí", "xưởng"]),
            ],
            policy=TopicGroupMatchPolicyIn(min_score=0.55, limit=2),
        )
    )
    assert len(result.matches) == 1
    assert result.matches[0].ref == "g:school"
    assert result.matches[0].score > 0.8
    # Distractor with weak semantic similarity did not score 1.0 and was excluded
    refs = [m.ref for m in result.matches]
    assert "g:distractor" not in refs


def test_case_d_semantic_paraphrase_discovered_without_exact_lexical() -> None:
    """Case D: Paraphrase without exact phrase match is discoverable via semantic evidence."""
    matcher = TopicGroupMatcher(
        MappingEmbeddingProvider(
            {
                "túi đeo lưng cho các em đi học đến trường": (0.85, 0.15),
                "balo học sinh": (0.90, 0.10),
                "phụ kiện thời trang": (0.2, 0.8),
            }
        )
    )
    result = matcher.match(
        TopicGroupMatchRequest(
            scope_ref="site:4",
            query="túi đeo lưng cho các em đi học đến trường",
            groups=[
                TopicGroupCandidateIn(ref="g:school", label="balo học sinh", examples=["balo học sinh"]),
                TopicGroupCandidateIn(ref="g:fashion", label="phụ kiện thời trang", examples=["phụ kiện thời trang"]),
            ],
            policy=TopicGroupMatchPolicyIn(min_score=0.55, limit=2),
        )
    )
    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.ref == "g:school"
    assert match.evidence.lexical_matched is False  # No lexical overlap
    assert match.evidence.semantic_score is not None and match.evidence.semantic_score >= 0.55
    assert match.score >= 0.55


def test_case_e_unrelated_content_returns_no_fabricated_matches() -> None:
    """Case E: Unrelated content yields no fabricated matches."""
    matcher = TopicGroupMatcher(
        MappingEmbeddingProvider(
            {
                "hướng dẫn nấu món phở bò truyền thống thơm ngon đậm đà": (0.0, 1.0),
                "balo học sinh": (1.0, 0.0),
                "chất liệu vải": (0.9, 0.1),
            }
        )
    )
    result = matcher.match(
        TopicGroupMatchRequest(
            scope_ref="site:4",
            query="hướng dẫn nấu món phở bò truyền thống thơm ngon đậm đà",
            groups=[
                TopicGroupCandidateIn(ref="g:school", label="balo học sinh", examples=["balo học sinh"]),
                TopicGroupCandidateIn(ref="g:fabric", label="chất liệu vải", examples=["chất liệu vải"]),
            ],
            policy=TopicGroupMatchPolicyIn(min_score=0.55, limit=3),
        )
    )
    assert len(result.matches) == 0


def test_case_f_long_article_captures_concept_outside_initial_window() -> None:
    """Case F: Relevant concept in a later paragraph is captured by segmentation."""
    # Construct a multi-paragraph article where paragraph 4 contains the key topic
    p0 = "Mở đầu bài viết giới thiệu chung về hoạt động doanh nghiệp tổng thể."
    p1 = "Các hoạt động quản trị nội bộ và quy trình hành chính văn phòng."
    p2 = "Phương thức tuyển dụng nhân sự và kế hoạch phát triển nguồn nhân lực."
    p3 = "Chiến lược tiếp thị và quảng bá thương hiệu trên các nền tảng số."
    p4 = "Chuyên đề đặc biệt về xưởng may balo học sinh cấp 2, cấp 3 uy tín chất lượng cao."
    
    long_article = f"{p0}\n\n{p1}\n\n{p2}\n\n{p3}\n\n{p4}"
    
    matcher = TopicGroupMatcher(
        MappingEmbeddingProvider(
            {
                p0: (0.1, 0.9),
                p1: (0.1, 0.9),
                p2: (0.1, 0.9),
                p3: (0.1, 0.9),
                p4: (0.95, 0.05),
                "balo học sinh": (1.0, 0.0),
            }
        )
    )
    result = matcher.match(
        TopicGroupMatchRequest(
            scope_ref="site:4",
            query=long_article,
            groups=[
                TopicGroupCandidateIn(ref="g:school", label="balo học sinh", examples=["balo học sinh"]),
            ],
            policy=TopicGroupMatchPolicyIn(min_score=0.50, limit=2),
        )
    )
    assert len(result.matches) == 1
    assert result.matches[0].ref == "g:school"
    assert result.matches[0].score >= 0.50


def test_case_g_site_isolation() -> None:
    """Case G: Only groups provided in request are returned; scope_ref preserved."""
    matcher = TopicGroupMatcher(
        MappingEmbeddingProvider(
            {
                "túi vải canvas": (1.0, 0.0),
                "túi vải canvas chất lượng cao": (0.9, 0.1),
            }
        )
    )
    req = TopicGroupMatchRequest(
        scope_ref="site:999",
        query="túi vải canvas chất lượng cao",
        groups=[
            TopicGroupCandidateIn(ref="site-999-group-1", label="túi vải canvas", examples=["túi vải canvas"]),
        ],
        policy=TopicGroupMatchPolicyIn(min_score=0.50, limit=2),
    )
    result = matcher.match(req)
    assert result.scope_ref == "site:999"
    assert len(result.matches) == 1
    assert result.matches[0].ref == "site-999-group-1"


def test_case_a_article_9598() -> None:
    """Case A: Article 9598 - chất liệu vải must not score 1.0; balo học sinh must rank at top."""
    lead = "Mỗi khi mùa khai giảng cận kề, nhu cầu tìm kiếm nguồn hàng cặp học sinh cấp 2, 3 chất lượng cao tăng vọt."
    sec_backpack = "Xưởng Gia Công May Cặp Học Sinh Cấp 2, 3 Uy Tín, Giá Gốc Tại Xưởng Hợp Phát."
    sec_fabric = "Chất Liệu Vải May Cao Cấp: Vải dù 1680D, vải polyester, vải canvas trượt nước tốt."
    sec_factory = "Quy trình sản xuất balo và máy móc hiện đại tại xưởng may uy tín."

    article = f"{lead}\n\n{sec_backpack}\n\n{sec_fabric}\n\n{sec_factory}"

    matcher = TopicGroupMatcher(
        MappingEmbeddingProvider(
            {
                lead: (0.8, 0.1, 0.0, 0.0),
                sec_backpack: (0.8, 0.2, 0.0, 0.0),
                sec_fabric: (0.1, 0.1, 0.7, 0.0),
                sec_factory: (0.3, 0.6, 0.0, 0.1),
                "balo học sinh": (0.8, 0.2, 0.0, 0.0),
                "sản xuất balo": (0.3, 0.6, 0.0, 0.1),
                "chất liệu vải": (0.0, 0.0, 0.8, 0.0),
            },
            default=(0.0, 0.0, 0.0, 1.0),
        )
    )
    result = matcher.match(
        TopicGroupMatchRequest(
            scope_ref="site:4",
            query=article,
            groups=[
                TopicGroupCandidateIn(ref="g-0016", label="balo học sinh", examples=["balo học sinh", "cặp học sinh"]),
                TopicGroupCandidateIn(ref="g-0013", label="sản xuất balo", examples=["sản xuất balo"]),
                TopicGroupCandidateIn(ref="g-0012", label="chất liệu vải", examples=["chất liệu vải", "vải dù"]),
            ],
            policy=TopicGroupMatchPolicyIn(min_score=0.50, limit=3),
        )
    )
    assert len(result.matches) >= 2
    refs = [m.ref for m in result.matches]
    assert refs[0] == "g-0016"  # balo học sinh ranks #1
    # 'chất liệu vải' does not dominate or receive 1.0
    fabric_match = next((m for m in result.matches if m.ref == "g-0012"), None)
    if fabric_match is not None:
        assert fabric_match.score < 0.80


def test_segment_content_helper() -> None:
    """Segment content splits long articles into bounded meaningful paragraphs and drops CTAs."""
    short_query = "balo học sinh"
    assert segment_content(short_query) == ["balo học sinh"]

    long_text = "\n\n".join([f"Đoạn văn thứ {i} với nội dung chi tiết mô tả quy trình sản xuất." for i in range(15)])
    segments = segment_content(long_text, max_segments=8)
    assert 1 < len(segments) <= 8

    # CTA filtering check
    mixed_text = (
        "Đoạn mở đầu giới thiệu bài viết cặp học sinh.\n"
        "📞Hotline tư vấn ngay: 0909 938 333| 📧Email:info.mayhopphat@gmail.com\n"
        "🚀Nhận mẫu vải miễn phí tại xưởng! Gọi ngay 0909 938 333 để đặt lịch.\n"
        "Đoạn kết luận về tiêu chuẩn chất lượng sản phẩm."
    )
    mixed_segs = segment_content(mixed_text, max_segments=10)
    assert len(mixed_segs) == 2
    assert "Hotline" not in mixed_segs[0] and "Hotline" not in mixed_segs[1]

