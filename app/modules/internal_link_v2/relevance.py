"""Article-level relevance for Internal Link V2.

Published targets are reused by content hash. The source article is embedded
only for the request and is not written to the vector store.
"""

from __future__ import annotations

import hashlib

from app.core.embedding.contracts import EmbeddingProvider
from app.core.text.normalization import normalize_text
from app.core.vector.contracts import VectorRecord, VectorStore
from app.modules.internal_link_v2.contracts import InternalLinkCandidateIn, InternalLinkRankRequest
from app.core.similarity.cosine import cosine_similarity

PUBLISHED_ARTICLE_NAMESPACE = "published_articles"


def apply_text_relevance(
    request: InternalLinkRankRequest,
    embedding: EmbeddingProvider,
    store: VectorStore | None = None,
) -> InternalLinkRankRequest:
    source = normalize_text(request.source_text)
    if source == "":
        return request
    if not embedding.is_loaded:
        embedding.load()

    source_vector = embedding.embed(source).vector
    updated: list[InternalLinkCandidateIn] = []
    for candidate in request.candidates:
        text = normalize_text(candidate.representation)
        if text == "":
            updated.append(candidate)
            continue
        vector = _target_vector(candidate.ref, text, embedding, store)
        cosine = cosine_similarity(source_vector, vector)
        relevance = max(0.0, min(1.0, cosine))
        updated.append(candidate.model_copy(update={"relevance": relevance}))
    return request.model_copy(update={"candidates": updated})


def _target_vector(
    ref: str,
    text: str,
    embedding: EmbeddingProvider,
    store: VectorStore | None,
) -> tuple[float, ...]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    model_key = embedding.model_key
    if store is not None:
        existing = store.get(PUBLISHED_ARTICLE_NAMESPACE, ref, model_key)
        if existing is not None and existing.content_hash == digest:
            return tuple(existing.embedding)
    result = embedding.embed(text)
    if store is not None:
        store.upsert(
            VectorRecord(
                namespace=PUBLISHED_ARTICLE_NAMESPACE,
                entity_ref=ref,
                content_hash=digest,
                model_key=result.model_key,
                embedding=result.vector,
                metadata={"kind": "article_representation"},
                dimensions=result.dimensions,
            )
        )
    return tuple(result.vector)
