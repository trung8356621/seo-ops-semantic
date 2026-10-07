from __future__ import annotations

import time
import uuid
from typing import Sequence

from app.core.embedding.contracts import EmbeddingProvider, EmbeddingResult
from app.core.text.concept_lexical import match_concept_lexically
from app.modules.concept_matching.contracts import (
    ConceptDefinitionIn,
    ConceptEntityOut,
    ConceptMatchAnalysisRequest,
    ConceptMatchAnalysisResponse,
    ConceptMatchDiagnostics,
    ConceptScoreOut,
    LexicalEvidenceOut,
)
from app.modules.concept_matching.decision import decide_suggested_match
from app.modules.concept_matching.scoring import ConceptScoreEvidence, score_entity_against_concept


class ConceptMatchAnalyzer:
    """Stateless concept matching: lexical evidence + optional batch semantic evidence.

    V1 does not persist embeddings or analysis rows. Laravel remains assignment authority.
    """

    def __init__(self, embedding: EmbeddingProvider) -> None:
        self._embedding = embedding

    def analyze(self, request: ConceptMatchAnalysisRequest) -> ConceptMatchAnalysisResponse:
        started = time.perf_counter()

        semantic_concepts = [c for c in request.concepts if c.needs_semantic()]
        lexical_only_concepts = [c for c in request.concepts if not c.needs_semantic()]

        unique_texts: list[str] = []
        seen: set[str] = set()
        if semantic_concepts:
            for entity in request.entities:
                if entity.text not in seen:
                    seen.add(entity.text)
                    unique_texts.append(entity.text)
            for concept in semantic_concepts:
                for example in concept.positive_examples:
                    if example not in seen:
                        seen.add(example)
                        unique_texts.append(example)
                for example in concept.negative_examples:
                    if example not in seen:
                        seen.add(example)
                        unique_texts.append(example)

        embed_started = time.perf_counter()
        vectors_by_text: dict[str, EmbeddingResult] = {}
        if unique_texts:
            vectors_by_text = self._embed_unique(unique_texts)
        embed_ms = int((time.perf_counter() - embed_started) * 1000)

        if vectors_by_text:
            sample = next(iter(vectors_by_text.values()))
            model_key = sample.model_key
            provider_key = sample.provider
            dimensions = sample.dimensions
        else:
            model_key = self._embedding.model_key or "none"
            provider_key = self._embedding.provider_key or "none"
            dimensions = int(self._embedding.dimensions or 0)

        score_started = time.perf_counter()
        entity_outs: list[ConceptEntityOut] = []
        for entity in request.entities:
            concept_scores: list[ConceptScoreOut] = []
            for concept in request.concepts:
                concept_scores.append(
                    self._score_pair(
                        entity_text=entity.text,
                        concept=concept,
                        vectors_by_text=vectors_by_text,
                        decision_policy=request.decision_policy,
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
                semantic_concepts=len(semantic_concepts),
                lexical_only_concepts=len(lexical_only_concepts),
            ),
        )

    def _score_pair(
        self,
        *,
        entity_text: str,
        concept: ConceptDefinitionIn,
        vectors_by_text: dict[str, EmbeddingResult],
        decision_policy,
    ) -> ConceptScoreOut:
        lexical_raw = match_concept_lexically(
            text=entity_text,
            positive_examples=concept.positive_examples,
            negative_examples=concept.negative_examples,
            match_mode=concept.match_mode,
        )
        lexical = LexicalEvidenceOut(
            matched=lexical_raw.matched,
            match_mode=lexical_raw.match_mode,
            matched_examples=list(lexical_raw.matched_examples),
            best_match=lexical_raw.best_match,
            negative_matched=lexical_raw.negative_matched,
            negative_matched_examples=list(lexical_raw.negative_matched_examples),
        )

        semantic: ConceptScoreEvidence | None = None
        if concept.needs_semantic():
            entity_vector = vectors_by_text[entity_text].vector
            pos_vecs = [vectors_by_text[t].vector for t in concept.positive_examples]
            neg_vecs = [vectors_by_text[t].vector for t in concept.negative_examples]
            semantic = score_entity_against_concept(
                entity_vector=entity_vector,
                positive_examples=concept.positive_examples,
                positive_vectors=pos_vecs,
                negative_examples=concept.negative_examples,
                negative_vectors=neg_vecs,
                decision_policy=None,  # strategy-aware decision applied below
            )

        suggested = decide_suggested_match(
            strategy=concept.matching_strategy,
            lexical=lexical,
            semantic=semantic,
            policy=decision_policy,
        )

        return ConceptScoreOut(
            key=concept.key,
            lexical=lexical,
            matching_strategy=concept.matching_strategy,
            positive_max=None if semantic is None else semantic.positive_max,
            positive_top_k_mean=None if semantic is None else semantic.positive_top_k_mean,
            negative_max=None if semantic is None else semantic.negative_max,
            margin=None if semantic is None else semantic.margin,
            best_positive_example=None if semantic is None else semantic.best_positive_example,
            best_positive_similarity=(
                None if semantic is None else semantic.best_positive_similarity
            ),
            best_negative_example=None if semantic is None else semantic.best_negative_example,
            best_negative_similarity=(
                None if semantic is None else semantic.best_negative_similarity
            ),
            suggested_match=suggested,
        )

    def _embed_unique(self, texts: Sequence[str]) -> dict[str, EmbeddingResult]:
        if not texts:
            raise ValueError("no texts to embed")
        results = self._embedding.embed_batch(list(texts))
        if len(results) != len(texts):
            raise ValueError("embed_batch returned unexpected length")
        return {text: result for text, result in zip(texts, results, strict=True)}
