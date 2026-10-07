from __future__ import annotations

import time
import uuid
from typing import Sequence

from app.core.embedding.contracts import EmbeddingProvider, EmbeddingResult
from app.modules.concept_matching.contracts import (
    ConceptEntityOut,
    ConceptMatchAnalysisRequest,
    ConceptMatchAnalysisResponse,
    ConceptMatchDiagnostics,
    ConceptScoreOut,
)
from app.modules.concept_matching.scoring import score_entity_against_concept


class ConceptMatchAnalyzer:
    """Stateless concept matching: batch-embed unique texts, score in memory.

    V1 does not persist embeddings or analysis rows. Laravel remains assignment authority.
    """

    def __init__(self, embedding: EmbeddingProvider) -> None:
        self._embedding = embedding

    def analyze(self, request: ConceptMatchAnalysisRequest) -> ConceptMatchAnalysisResponse:
        started = time.perf_counter()

        unique_texts: list[str] = []
        seen: set[str] = set()
        for entity in request.entities:
            if entity.text not in seen:
                seen.add(entity.text)
                unique_texts.append(entity.text)
        for concept in request.concepts:
            for example in concept.positive_examples:
                if example not in seen:
                    seen.add(example)
                    unique_texts.append(example)
            for example in concept.negative_examples:
                if example not in seen:
                    seen.add(example)
                    unique_texts.append(example)

        embed_started = time.perf_counter()
        vectors_by_text = self._embed_unique(unique_texts)
        embed_ms = int((time.perf_counter() - embed_started) * 1000)

        sample = next(iter(vectors_by_text.values()))
        model_key = sample.model_key
        provider_key = sample.provider
        dimensions = sample.dimensions

        score_started = time.perf_counter()
        entity_outs: list[ConceptEntityOut] = []
        for entity in request.entities:
            entity_vector = vectors_by_text[entity.text].vector
            concept_scores: list[ConceptScoreOut] = []
            for concept in request.concepts:
                pos_vecs = [vectors_by_text[t].vector for t in concept.positive_examples]
                neg_vecs = [vectors_by_text[t].vector for t in concept.negative_examples]
                evidence = score_entity_against_concept(
                    entity_vector=entity_vector,
                    positive_examples=concept.positive_examples,
                    positive_vectors=pos_vecs,
                    negative_examples=concept.negative_examples,
                    negative_vectors=neg_vecs,
                    decision_policy=request.decision_policy,
                )
                concept_scores.append(
                    ConceptScoreOut(
                        key=concept.key,
                        positive_max=evidence.positive_max,
                        positive_top_k_mean=evidence.positive_top_k_mean,
                        negative_max=evidence.negative_max,
                        margin=evidence.margin,
                        best_positive_example=evidence.best_positive_example,
                        best_positive_similarity=evidence.best_positive_similarity,
                        best_negative_example=evidence.best_negative_example,
                        best_negative_similarity=evidence.best_negative_similarity,
                        suggested_match=evidence.suggested_match,
                    )
                )
            entity_outs.append(
                ConceptEntityOut(ref=entity.ref, text=entity.text, concepts=concept_scores)
            )
        score_ms = int((time.perf_counter() - score_started) * 1000)
        total_ms = int((time.perf_counter() - started) * 1000)

        return ConceptMatchAnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            scope_ref=request.scope_ref,
            language=request.language,
            entities=entity_outs,
            diagnostics=ConceptMatchDiagnostics(
                entity_count=len(request.entities),
                concept_count=len(request.concepts),
                unique_text_count=len(unique_texts),
                embed_ms=embed_ms,
                score_ms=score_ms,
                total_ms=total_ms,
                model=model_key,
                provider=provider_key,
                dimensions=dimensions,
                cache="direct_embed_batch",
            ),
        )

    def _embed_unique(self, texts: Sequence[str]) -> dict[str, EmbeddingResult]:
        if not texts:
            raise ValueError("no texts to embed")
        results = self._embedding.embed_batch(list(texts))
        if len(results) != len(texts):
            raise ValueError("embed_batch returned unexpected length")
        return {text: result for text, result in zip(texts, results, strict=True)}
