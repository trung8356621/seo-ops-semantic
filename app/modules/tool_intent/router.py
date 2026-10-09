"""Closed-set tool intent router.

Semantic evidence only. Callers decide execution.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.embedding.contracts import EmbeddingProvider
from app.modules.concept_matching.analyzer import ConceptMatchAnalyzer
from app.modules.concept_matching.contracts import (
    ConceptDefinitionIn,
    ConceptEntityIn,
    ConceptMatchAnalysisRequest,
    ConceptScoreOut,
)
from app.core.similarity.cosine import cosine_similarity
from app.modules.tool_intent.catalog import TOOL_INTENT_NAMESPACE, builtin_intents
from app.modules.tool_intent.weighted import WeightedGroup, WeightedTarget, group_relevance, rank_weighted
from app.modules.tool_intent.contracts import (
    ToolIntentDefinitionIn,
    ToolIntentEvidenceOut,
    ToolIntentMatchOut,
    ToolIntentMatchRequest,
    ToolIntentMatchResponse,
    ToolIntentPolicyIn,
    ToolIntentStatus,
    WeightedCandidateOut,
    WeightedMatchRequest,
    WeightedMatchResponse,
)


@dataclass(frozen=True)
class _RankedIntent:
    key: str
    semantic_score: float | None
    lexical_matched: bool
    best_example: str | None


class ToolIntentRouter:
    def __init__(self, embedding: EmbeddingProvider) -> None:
        self._embedding = embedding
        self._analyzer = ConceptMatchAnalyzer(embedding)

    def match(self, request: ToolIntentMatchRequest) -> ToolIntentMatchResponse:
        intents = self._resolve_intents(request)
        if not intents:
            return ToolIntentMatchResponse(
                namespace=TOOL_INTENT_NAMESPACE,
                status="none",
                scope_ref=request.scope_ref,
                matches=[],
                policy=request.policy,
            )

        analysis = self._analyzer.analyze(
            ConceptMatchAnalysisRequest(
                scope_ref=request.scope_ref,
                language=request.language,
                entities=[ConceptEntityIn(ref="query", text=request.query)],
                concepts=[
                    ConceptDefinitionIn(
                        key=item.key,
                        positive_examples=item.positive_examples,
                        negative_examples=item.negative_examples,
                        match_mode="phrase",
                        matching_strategy="hybrid",
                    )
                    for item in intents
                ],
                decision_policy=None,
            )
        )
        scores = analysis.entities[0].concepts
        ranked = [_rank(score) for score in scores]
        status, ordered = _classify(ranked, request.policy)
        return ToolIntentMatchResponse(
            namespace=TOOL_INTENT_NAMESPACE,
            status=status,
            scope_ref=request.scope_ref,
            matches=[_to_match(item, ordered) for item in ordered],
            policy=request.policy,
        )

    def _resolve_intents(self, request: ToolIntentMatchRequest) -> list[ToolIntentDefinitionIn]:
        if request.intents:
            if request.allowed_keys is None:
                return list(request.intents)
            allowed = set(request.allowed_keys)
            return [item for item in request.intents if item.key in allowed]

        rows = builtin_intents(request.allowed_keys)
        return [ToolIntentDefinitionIn.model_validate(row) for row in rows]

    def match_weighted(self, request: WeightedMatchRequest) -> WeightedMatchResponse:
        groups = [
            WeightedGroup(
                id=group.id,
                examples=group.examples,
                targets=[WeightedTarget(ref=target.ref, weight=target.weight) for target in group.targets],
                enabled=group.enabled,
            )
            for group in request.groups
            if group.enabled
        ]
        texts = [request.query]
        spans: list[tuple[str, int, int]] = []
        for group in groups:
            start = len(texts)
            texts.extend(group.examples)
            spans.append((group.id, start, len(texts)))
        if not self._embedding.is_loaded:
            self._embedding.load()
        vectors = [item.vector for item in self._embedding.embed_batch(texts)]
        query_vector = vectors[0]
        relevances: dict[str, float] = {}
        examples: dict[str, str] = {}
        group_by_id = {group.id: group for group in groups}
        for group_id, start, end in spans:
            sims = [cosine_similarity(query_vector, vectors[index]) for index in range(start, end)]
            relevances[group_id] = group_relevance(sims)
            best_index = max(range(len(sims)), key=lambda index: sims[index]) if sims else 0
            examples[group_id] = group_by_id[group_id].examples[best_index] if sims else ""
        status, winner, ranked = rank_weighted(
            relevances,
            examples,
            groups,
            floor=request.policy.min_positive_score,
            margin=request.policy.min_margin,
        )
        return WeightedMatchResponse(
            status=status,
            winner=winner,
            candidates=[
                WeightedCandidateOut(
                    ref=item.ref,
                    semantic_relevance=item.semantic_relevance,
                    weight=item.weight,
                    score=item.score,
                    group_id=item.group_id,
                    example=item.example,
                )
                for item in ranked
            ],
            policy=request.policy,
        )


def _rank(score: ConceptScoreOut) -> _RankedIntent:
    lexical_hit = score.lexical.matched and not score.lexical.negative_matched
    return _RankedIntent(
        key=score.key,
        semantic_score=score.positive_max,
        lexical_matched=lexical_hit,
        best_example=score.lexical.best_match or score.best_positive_example,
    )


def _classify(
    ranked: list[_RankedIntent],
    policy: ToolIntentPolicyIn,
) -> tuple[ToolIntentStatus, list[_RankedIntent]]:
    lexical_hits = [item for item in ranked if item.lexical_matched]
    by_semantic = sorted(
        ranked,
        key=lambda item: item.semantic_score if item.semantic_score is not None else -1.0,
        reverse=True,
    )

    if len(lexical_hits) > 1:
        return "ambiguous", lexical_hits

    if len(lexical_hits) == 1:
        chosen = lexical_hits[0]
        top = by_semantic[0]
        if (
            top.key != chosen.key
            and top.semantic_score is not None
            and top.semantic_score >= policy.min_positive_score
            and _margin(top.semantic_score, chosen.semantic_score) >= policy.min_margin
        ):
            return "ambiguous", [chosen, top]
        return "confident", [chosen, *[item for item in by_semantic if item.key != chosen.key]]

    eligible = [
        item
        for item in by_semantic
        if item.semantic_score is not None and item.semantic_score >= policy.min_positive_score
    ]
    if not eligible:
        return "none", by_semantic[:2]

    top = eligible[0]
    second = eligible[1] if len(eligible) > 1 else None
    if second is not None and _margin(top.semantic_score, second.semantic_score) < policy.min_margin:
        return "ambiguous", [top, second]
    rest = [item for item in by_semantic if item.key != top.key]
    return "confident", [top, *rest]


def _margin(top: float | None, second: float | None) -> float:
    if top is None:
        return 0.0
    if second is None:
        return top
    return top - second


def _to_match(item: _RankedIntent, ordered: list[_RankedIntent]) -> ToolIntentMatchOut:
    top_score = ordered[0].semantic_score
    margin = None
    if len(ordered) > 1:
        margin = _margin(item.semantic_score, ordered[1].semantic_score if item is ordered[0] else top_score)
    score = 1.0 if item.lexical_matched and item is ordered[0] else (item.semantic_score or 0.0)
    return ToolIntentMatchOut(
        ref=item.key,
        score=score,
        evidence=ToolIntentEvidenceOut(
            lexical_matched=item.lexical_matched,
            semantic_score=item.semantic_score,
            runner_up_margin=margin if item is ordered[0] else None,
            best_positive_example=item.best_example,
        ),
    )
