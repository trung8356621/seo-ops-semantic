from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from app.config import Settings
from app.core.clustering import ClusterPoint
from app.core.embedding.contracts import EmbeddingProvider, EmbeddingResult
from app.core.text.normalization import normalize_text
from app.core.vector.contracts import VectorRecord, VectorStore
from app.modules.keyword_grouping.contracts import (
    KeywordGroupAnalysisRequest,
    KeywordGroupAnalysisResponse,
    KeywordGroupDiagnostics,
    KeywordIn,
)
from app.modules.keyword_grouping.grouping import cluster_points, materialize_groups_with_scores
from app.modules.keyword_grouping.hybrid import run_hybrid_semantic_lexical_v1
from app.modules.keyword_grouping.repository import KeywordGroupAnalysisRepository


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def compute_input_hash(scope_ref: str, language: str | None, keywords: list[KeywordIn]) -> str:
    payload = {
        "scope_ref": scope_ref,
        "language": language,
        "keywords": [{"ref": k.ref, "text": normalize_text(k.text)} for k in sorted(keywords, key=lambda x: x.ref)],
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def grouping_namespace(scope_ref: str) -> str:
    return f"keyword-group:{scope_ref}"


def model_cache_key(provider: EmbeddingProvider) -> str:
    return f"{provider.provider_key}:{provider.model_key}:{provider.model_version}"


class KeywordGroupAnalyzer:
    """Low-level keyword → embedding → semantic group orchestrator.

    No Topic business concepts. Reuses core clustering + embedding cache.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        embedding: EmbeddingProvider,
        vectors: VectorStore | None = None,
        repository: KeywordGroupAnalysisRepository | None = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding
        self._vectors = vectors
        self._repository = repository

    def analyze(self, request: KeywordGroupAnalysisRequest) -> KeywordGroupAnalysisResponse:
        t0 = time.perf_counter()
        analysis_id = str(uuid.uuid4())
        prepared = self._prepare_keywords(request.keywords)
        computed_hash = compute_input_hash(
            request.scope_ref,
            request.language,
            prepared,
        )
        if request.input_hash is not None and request.input_hash != "":
            if request.input_hash != computed_hash:
                raise ValueError(
                    "input_hash mismatch: client hash does not match canonical "
                    "hash of normalized payload"
                )
        input_hash = computed_hash

        embed_t0 = time.perf_counter()
        embeddings, cache_stats = self._embed_keywords(request.scope_ref, prepared)
        embed_ms = int((time.perf_counter() - embed_t0) * 1000)

        points = [
            ClusterPoint(ref=item.ref, vector=embeddings[item.ref].vector)
            for item in prepared
        ]
        texts = {item.ref: normalize_text(item.text) for item in prepared}
        cluster_t0 = time.perf_counter()
        algorithm = (self._settings.keyword_group_algorithm or "").strip().lower()
        hybrid_diag = None
        if algorithm in {"hybrid_semantic_lexical_v1", "hybrid_v1", "hybrid"}:
            groups, unassigned, hybrid_diag, algo_config = run_hybrid_semantic_lexical_v1(
                points=points,
                texts_by_ref=texts,
                settings=self._settings,
            )
            algorithm_name = hybrid_diag.strategy
        else:
            # Escape hatch: legacy cosine clustering for keyword-groups.
            cluster_result = cluster_points(points, self._settings)
            score_to_rep = self._scores_to_representative(points, cluster_result)
            groups, unassigned, _low_conf = materialize_groups_with_scores(
                cluster_result=cluster_result,
                texts_by_ref=texts,
                score_to_rep=score_to_rep,
                settings=self._settings,
            )
            algorithm_name = cluster_result.algorithm
            algo_config = dict(cluster_result.config)
        cluster_ms = int((time.perf_counter() - cluster_t0) * 1000)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        response = KeywordGroupAnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            scope_ref=request.scope_ref,
            language=request.language,
            input_hash=input_hash,
            request_id=request.request_id,
            groups=groups,
            unassigned=unassigned,
            diagnostics=KeywordGroupDiagnostics(
                keyword_count=len(prepared),
                group_count=len(groups),
                unassigned_count=len(unassigned),
                algorithm=algorithm_name,
                algorithm_config=algo_config,
                timings_ms={
                    "embed_ms": embed_ms,
                    "cluster_ms": cluster_ms,
                    "total_ms": duration_ms,
                },
                embedding_cache=cache_stats,
                strategy=hybrid_diag.strategy if hybrid_diag else None,
                lexical_reject_count=hybrid_diag.lexical_reject_count if hybrid_diag else None,
                rescue_assignment_count=(
                    hybrid_diag.rescue_assignment_count if hybrid_diag else None
                ),
                ambiguous_count=hybrid_diag.ambiguous_count if hybrid_diag else None,
            ),
        )

        if self._repository is not None:
            self._repository.save(response)

        return response

    def _prepare_keywords(self, keywords: list[KeywordIn]) -> list[KeywordIn]:
        if len(keywords) > self._settings.topic_max_keywords:
            raise ValueError(
                f"keyword count {len(keywords)} exceeds TOPIC_MAX_KEYWORDS="
                f"{self._settings.topic_max_keywords}"
            )
        prepared: list[KeywordIn] = []
        for item in keywords:
            text = normalize_text(item.text)
            if text == "":
                raise ValueError(f"keyword ref={item.ref} has empty text after normalization")
            if len(text) > self._settings.topic_max_text_length:
                raise ValueError(
                    f"keyword ref={item.ref} text length {len(text)} exceeds "
                    f"TOPIC_MAX_TEXT_LENGTH={self._settings.topic_max_text_length}"
                )
            prepared.append(KeywordIn(ref=item.ref, text=text))
        return sorted(prepared, key=lambda row: row.ref)

    def _embed_keywords(
        self,
        scope_ref: str,
        keywords: list[KeywordIn],
    ) -> tuple[dict[str, EmbeddingResult], dict[str, int]]:
        namespace = grouping_namespace(scope_ref)
        model_key = model_cache_key(self._embedding)
        cache_hits = 0
        cache_misses = 0
        results: dict[str, EmbeddingResult] = {}
        to_embed: list[KeywordIn] = []

        for item in keywords:
            digest = content_hash(item.text)
            cached: EmbeddingResult | None = None
            if self._settings.topic_embedding_cache_enabled and self._vectors is not None:
                record = self._vectors.get(namespace, item.ref, model_key)
                if (
                    record is not None
                    and record.content_hash == digest
                    and record.dimensions > 0
                ):
                    cached = EmbeddingResult(
                        vector=tuple(float(x) for x in record.embedding),
                        model_key=self._embedding.model_key,
                        model_version=self._embedding.model_version,
                        dimensions=record.dimensions,
                        provider=self._embedding.provider_key,
                    )
            if cached is not None:
                results[item.ref] = cached
                cache_hits += 1
            else:
                to_embed.append(item)
                cache_misses += 1

        if to_embed:
            batch = self._embedding.embed_batch([item.text for item in to_embed])
            for item, emb in zip(to_embed, batch, strict=True):
                results[item.ref] = emb
                if self._settings.topic_embedding_cache_enabled and self._vectors is not None:
                    self._vectors.upsert(
                        VectorRecord(
                            namespace=namespace,
                            entity_ref=item.ref,
                            content_hash=content_hash(item.text),
                            model_key=model_key,
                            embedding=emb.vector,
                            metadata={
                                "model_name": emb.model_key,
                                "model_version": emb.model_version,
                                "provider": emb.provider,
                            },
                            dimensions=emb.dimensions,
                        )
                    )

        return results, {"hits": cache_hits, "misses": cache_misses}

    def _scores_to_representative(
        self,
        points: list[ClusterPoint],
        cluster_result,
    ) -> dict[str, dict[str, float]]:
        by_ref = {p.ref: np.asarray(p.vector, dtype=np.float64) for p in points}
        out: dict[str, dict[str, float]] = {}
        for cluster in cluster_result.groups:
            rep = by_ref[cluster.representative_ref]
            rep_norm = float(np.linalg.norm(rep)) or 1.0
            rep_u = rep / rep_norm
            scores: dict[str, float] = {}
            for ref in cluster.member_refs:
                vec = by_ref[ref]
                norm = float(np.linalg.norm(vec)) or 1.0
                scores[ref] = float(np.dot(rep_u, vec / norm))
            out[cluster.group_key] = scores
        return out
