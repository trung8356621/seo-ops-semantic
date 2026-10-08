"""Maintainable example catalog for namespace ``agent_tool_intents``.

Callers may send their own examples. This catalog is the default closed set,
not a hardcoded decision threshold.
"""

from __future__ import annotations

TOOL_INTENT_NAMESPACE = "agent_tool_intents"

# Per-use-case gates. Cosine scores are evidence, not probabilities.
DEFAULT_MIN_POSITIVE_SCORE = 0.62
DEFAULT_MIN_MARGIN = 0.08

BUILTIN_INTENTS: dict[str, dict[str, list[str]]] = {
    "seo_audit.worst_articles": {
        "positive_examples": [
            "tìm bài seo kém",
            "bài nào cần tối ưu",
            "tìm bài điểm thấp",
            "nội dung nào đang hoạt động tệ",
            "worst seo articles",
        ],
        "negative_examples": [
            "xuất bản bài viết",
            "publish this article",
        ],
    },
    "gsc.performance": {
        "positive_examples": [
            "traffic tháng này",
            "impression giảm",
            "click gsc",
            "query tụt",
            "hiệu suất tìm kiếm",
            "search console performance",
        ],
        "negative_examples": [],
    },
    "content_projects.read": {
        "positive_examples": [
            "draft hiện tại",
            "project tháng này",
            "kế hoạch nội dung",
            "bài đang chờ",
            "content project status",
        ],
        "negative_examples": [],
    },
    "articles.inventory": {
        "positive_examples": [
            "danh sách bài viết",
            "kho bài trên site",
            "article inventory",
        ],
        "negative_examples": [],
    },
    "links.internal": {
        "positive_examples": [
            "internal link nào đang có",
            "liên kết nội bộ",
            "internal links inventory",
        ],
        "negative_examples": [],
    },
    "keywords.landscape": {
        "positive_examples": [
            "từ khóa đang theo dõi",
            "keyword landscape",
            "phủ keyword",
        ],
        "negative_examples": [],
    },
}


def builtin_intents(allowed_keys: list[str] | None = None) -> list[dict[str, object]]:
    keys = list(BUILTIN_INTENTS if allowed_keys is None else allowed_keys)
    out: list[dict[str, object]] = []
    for key in keys:
        row = BUILTIN_INTENTS.get(key)
        if row is None:
            continue
        out.append(
            {
                "key": key,
                "positive_examples": list(row["positive_examples"]),
                "negative_examples": list(row["negative_examples"]),
            }
        )
    return out
