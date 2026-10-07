from __future__ import annotations

from app.modules.keyword_grouping.contracts import (
    GroupMemberOut,
    GroupOut,
    KeywordGroupAnalysisResponse,
    KeywordGroupDiagnostics,
    UnassignedOut,
)
from app.modules.keyword_grouping.repository import KeywordGroupAnalysisRepository


class _FakeCursor:
    def __init__(self, row: dict | None) -> None:
        self._row = row
        self.rowcount = 1 if row is not None else 0
        self.executed: list[tuple] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params=None) -> None:  # noqa: ANN001
        self.executed.append((sql, params))

    def fetchone(self) -> dict | None:
        return self._row


class _FakeConn:
    def __init__(self, row: dict | None) -> None:
        self._row = row
        self.commits = 0

    def cursor(self, row_factory=None):  # noqa: ANN001
        return _FakeCursor(self._row)

    def commit(self) -> None:
        self.commits += 1


def _sample_response() -> KeywordGroupAnalysisResponse:
    return KeywordGroupAnalysisResponse(
        analysis_id="a-1",
        status="completed",
        scope_ref="site-4",
        language="vi",
        input_hash="abc",
        groups=[
            GroupOut(
                group_ref="g-0001",
                representative_ref="1",
                representative_text="balo học sinh",
                member_count=1,
                mean_similarity=1.0,
                min_similarity=1.0,
                cohesion=1.0,
                members=[
                    GroupMemberOut(
                        ref="1",
                        text="balo học sinh",
                        similarity_score=1.0,
                        is_representative=True,
                    )
                ],
            )
        ],
        unassigned=[
            UnassignedOut(ref="99", text="vali", reason="below_threshold_or_small_component")
        ],
        diagnostics=KeywordGroupDiagnostics(
            keyword_count=2,
            group_count=1,
            unassigned_count=1,
            algorithm="cosine_average_linkage_v1",
            algorithm_config={},
            timings_ms={"total_ms": 12},
            embedding_cache={"hits": 0, "misses": 2},
        ),
    )


def test_get_returns_keyword_group_payload() -> None:
    response = _sample_response()
    conn = _FakeConn({"result_json": response.model_dump()})
    loaded = KeywordGroupAnalysisRepository(conn).get("a-1")  # type: ignore[arg-type]
    assert loaded is not None
    assert loaded.scope_ref == "site-4"
    assert loaded.groups[0].representative_ref == "1"
    assert "suggested_label" not in loaded.model_dump()


def test_get_ignores_topic_shaped_payload() -> None:
    conn = _FakeConn(
        {
            "result_json": {
                "analysis_id": "t-1",
                "status": "completed",
                "site_ref": "6",
                "language": "vi",
                "input_hash": "x",
                "groups": [],
                "unassigned": [],
            }
        }
    )
    assert KeywordGroupAnalysisRepository(conn).get("t-1") is None  # type: ignore[arg-type]


def test_delete_only_keyword_group_rows() -> None:
    response = _sample_response()
    conn = _FakeConn({"result_json": response.model_dump()})
    repo = KeywordGroupAnalysisRepository(conn)  # type: ignore[arg-type]
    assert repo.delete("a-1") is True
    assert conn.commits == 1
