from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.modules.topic.contracts import TopicAnalysisResponse


class TopicAnalysisRepository:
    """Persist disposable Topic *analysis* runs (not business Topics)."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def save(self, response: TopicAnalysisResponse) -> None:
        payload = response.model_dump()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_analysis_runs (
                    analysis_id, site_ref, language, input_hash, status, request_id,
                    model_provider, model_name, model_version, model_dimensions,
                    algorithm, algorithm_config, keyword_count, group_count,
                    unassigned_count, singleton_count, diagnostics, result_json,
                    error, started_at, finished_at, duration_ms
                ) VALUES (
                    %(analysis_id)s, %(site_ref)s, %(language)s, %(input_hash)s, %(status)s,
                    %(request_id)s, %(model_provider)s, %(model_name)s, %(model_version)s,
                    %(model_dimensions)s, %(algorithm)s, %(algorithm_config)s,
                    %(keyword_count)s, %(group_count)s, %(unassigned_count)s,
                    %(singleton_count)s, %(diagnostics)s, %(result_json)s, %(error)s,
                    %(started_at)s, %(finished_at)s, %(duration_ms)s
                )
                ON CONFLICT (analysis_id) DO UPDATE SET
                    result_json = EXCLUDED.result_json,
                    diagnostics = EXCLUDED.diagnostics,
                    status = EXCLUDED.status
                """,
                {
                    "analysis_id": response.analysis_id,
                    "site_ref": response.site_ref,
                    "language": response.language,
                    "input_hash": response.input_hash,
                    "status": response.status,
                    "request_id": response.request_id,
                    "model_provider": response.model.provider,
                    "model_name": response.model.name,
                    "model_version": response.model.version,
                    "model_dimensions": response.model.dimensions,
                    "algorithm": response.diagnostics.algorithm,
                    "algorithm_config": Jsonb(response.diagnostics.algorithm_config),
                    "keyword_count": response.diagnostics.keyword_count,
                    "group_count": response.diagnostics.group_count,
                    "unassigned_count": response.diagnostics.unassigned_count,
                    "singleton_count": response.diagnostics.singleton_count,
                    "diagnostics": Jsonb(response.diagnostics.model_dump()),
                    "result_json": Jsonb(payload),
                    "error": response.error,
                    "started_at": response.started_at,
                    "finished_at": response.finished_at,
                    "duration_ms": response.duration_ms,
                },
            )

            cur.execute(
                "DELETE FROM topic_analysis_members WHERE analysis_id = %s",
                (response.analysis_id,),
            )
            cur.execute(
                "DELETE FROM topic_analysis_groups WHERE analysis_id = %s",
                (response.analysis_id,),
            )

            for group in response.groups:
                cur.execute(
                    """
                    INSERT INTO topic_analysis_groups (
                        analysis_id, group_ref, suggested_label, member_count,
                        mean_similarity, min_similarity, cohesion, payload
                    ) VALUES (
                        %(analysis_id)s, %(group_ref)s, %(suggested_label)s, %(member_count)s,
                        %(mean_similarity)s, %(min_similarity)s, %(cohesion)s, %(payload)s
                    )
                    """,
                    {
                        "analysis_id": response.analysis_id,
                        "group_ref": group.group_ref,
                        "suggested_label": group.suggested_label,
                        "member_count": group.member_count,
                        "mean_similarity": group.mean_similarity,
                        "min_similarity": group.min_similarity,
                        "cohesion": group.cohesion,
                        "payload": Jsonb(group.model_dump()),
                    },
                )
                for member in group.members:
                    cur.execute(
                        """
                        INSERT INTO topic_analysis_members (
                            analysis_id, group_ref, keyword_ref, text,
                            similarity_score, confidence, is_representative
                        ) VALUES (
                            %(analysis_id)s, %(group_ref)s, %(keyword_ref)s, %(text)s,
                            %(similarity_score)s, %(confidence)s, %(is_representative)s
                        )
                        """,
                        {
                            "analysis_id": response.analysis_id,
                            "group_ref": group.group_ref,
                            "keyword_ref": member.keyword_ref,
                            "text": member.text,
                            "similarity_score": member.similarity_score,
                            "confidence": member.confidence,
                            "is_representative": member.is_representative,
                        },
                    )
        self._conn.commit()

    def get(self, analysis_id: str) -> TopicAnalysisResponse | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT result_json FROM topic_analysis_runs WHERE analysis_id = %s",
                (analysis_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        payload: dict[str, Any] = row["result_json"]
        return TopicAnalysisResponse.model_validate(payload)

    def delete(self, analysis_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM topic_analysis_runs WHERE analysis_id = %s",
                (analysis_id,),
            )
            deleted = cur.rowcount
        self._conn.commit()
        return deleted > 0
