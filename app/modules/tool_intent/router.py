"""Closed-set tool intent router.

Semantic evidence only. Callers decide execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

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
    HybridMatchRequest,
    HybridMatchResponse,
    HybridCandidateOut,
    HybridOperationOut,
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

    def match_hybrid(self, request: HybridMatchRequest) -> HybridMatchResponse:
        global_groups = _weighted_groups(request.global_groups)
        module_groups = {key: _weighted_groups(value) for key, value in request.modules.items()}
        all_groups = [*global_groups, *[group for groups in module_groups.values() for group in groups]]
        relevances, examples = self._embed_groups(request.query, all_groups)
        _, _, global_ranked = rank_weighted(relevances, examples, global_groups, floor=-1.0, margin=0.0)
        lexical = _lexical_evidence(request.query, request.lexical_hints, request.policy.lexical_ceiling)
        candidates = []
        for item in global_ranked:
            bonus, matched = lexical.get(item.ref, (0.0, []))
            if item.semantic_relevance < request.policy.min_semantic_candidate:
                continue
            candidates.append(HybridCandidateOut(
                ref=item.ref, semantic_score=item.score, lexical_bonus=bonus,
                combined_score=min(1.0, item.score + bonus), matched_lexical_groups=matched,
                group_id=item.group_id, example=item.example,
            ))
        candidates.sort(key=lambda item: item.combined_score, reverse=True)
        candidates = candidates[:request.policy.max_modules]
        operations = []
        for candidate in candidates:
            groups = module_groups.get(candidate.ref, [])
            _, _, ranked = rank_weighted(relevances, examples, groups, floor=-1.0, margin=0.0)
            for item in ranked:
                if item.semantic_relevance < request.policy.min_operation_score:
                    continue
                operations.append(HybridOperationOut(
                    module=candidate.ref, operation=item.ref,
                    global_combined_score=candidate.combined_score,
                    internal_semantic_score=item.score,
                    final_score=(request.policy.global_coefficient * candidate.combined_score)
                    + (request.policy.internal_coefficient * item.score),
                    group_id=item.group_id, example=item.example,
                ))
        operations.sort(key=lambda item: item.final_score, reverse=True)
        if not operations:
            reason = "no_supported_operation" if candidates else "no_relevant_module"
            return HybridMatchResponse(status="unsupported" if candidates else "none", global_candidates=candidates,
                operation_candidates=[], reason=reason, policy=request.policy)
        if len(operations) > 1 and operations[0].final_score - operations[1].final_score < request.policy.final_margin:
            return HybridMatchResponse(status="ambiguous", global_candidates=candidates,
                operation_candidates=operations[:6], reason="final_operation_margin", policy=request.policy)
        winner = operations[0]
        return HybridMatchResponse(status="confident", module=winner.module, operation=winner.operation,
            global_candidates=candidates, operation_candidates=operations[:6], reason="final_operation_selected", policy=request.policy)

    def _embed_groups(self, query: str, groups: list[WeightedGroup]) -> tuple[dict[str, float], dict[str, str]]:
        texts = [query]
        spans = []
        for group in groups:
            start = len(texts); texts.extend(group.examples); spans.append((group, start, len(texts)))
        if not self._embedding.is_loaded:
            self._embedding.load()
        vectors = [item.vector for item in self._embedding.embed_batch(texts)]
        relevances, examples = {}, {}
        for group, start, end in spans:
            sims = [cosine_similarity(vectors[0], vectors[index]) for index in range(start, end)]
            relevances[group.id] = group_relevance(sims)
            examples[group.id] = group.examples[max(range(len(sims)), key=lambda index: sims[index])] if sims else ""
        return relevances, examples


def _weighted_groups(raw_groups) -> list[WeightedGroup]:
    return [WeightedGroup(id=g.id, examples=g.examples,
        targets=[WeightedTarget(ref=t.ref, weight=t.weight) for t in g.targets], enabled=g.enabled)
        for g in raw_groups if g.enabled]


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).casefold(), flags=re.UNICODE).split())


def _lexical_evidence(query: str, hints, ceiling: float) -> dict[str, tuple[float, list[str]]]:
    normalised = f" {_normalise(query)} "
    by_module: dict[str, tuple[float, list[str]]] = {}
    for hint in hints:
        if not hint.enabled:
            continue
        if not any(f" {_normalise(phrase)} " in normalised for phrase in hint.phrases):
            continue
        score, groups = by_module.get(hint.module, (0.0, []))
        by_module[hint.module] = (min(ceiling, score + hint.weight), [*groups, hint.id])
    return by_module


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
