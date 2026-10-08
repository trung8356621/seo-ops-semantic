from __future__ import annotations

from app.core.embedding.contracts import EmbeddingProvider
from app.modules.concept_matching.analyzer import ConceptMatchAnalyzer
from app.modules.concept_matching.contracts import (
    ConceptDefinitionIn,
    ConceptEntityIn,
    ConceptMatchAnalysisRequest,
)
from app.modules.topic_group_retrieval.contracts import (
    TopicGroupEvidenceOut,
    TopicGroupMatchOut,
    TopicGroupMatchRequest,
    TopicGroupMatchResponse,
)


class TopicGroupMatcher:
    def __init__(self, embedding: EmbeddingProvider) -> None:
        self._analyzer = ConceptMatchAnalyzer(embedding)

    def match(self, request: TopicGroupMatchRequest) -> TopicGroupMatchResponse:
        analysis = self._analyzer.analyze(
            ConceptMatchAnalysisRequest(
                scope_ref=request.scope_ref,
                language=request.language,
                entities=[ConceptEntityIn(ref="query", text=request.query)],
                concepts=[
                    ConceptDefinitionIn(
                        key=group.ref,
                        positive_examples=_examples(group.label, group.examples),
                        negative_examples=[],
                        match_mode="phrase",
                        matching_strategy="hybrid",
                    )
                    for group in request.groups
                ],
                decision_policy=None,
            )
        )
        ranked: list[TopicGroupMatchOut] = []
        for score in analysis.entities[0].concepts:
            semantic = score.positive_max
            lexical = score.lexical.matched and not score.lexical.negative_matched
            rank_score = 1.0 if lexical else (semantic if semantic is not None else 0.0)
            if not lexical and (semantic is None or semantic < request.policy.min_score):
                continue
            ranked.append(
                TopicGroupMatchOut(
                    ref=score.key,
                    score=rank_score,
                    evidence=TopicGroupEvidenceOut(
                        lexical_matched=lexical,
                        semantic_score=semantic,
                        best_example=score.lexical.best_match or score.best_positive_example,
                    ),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return TopicGroupMatchResponse(
            scope_ref=request.scope_ref,
            matches=ranked[: request.policy.limit],
        )


def _examples(label: str, examples: list[str]) -> list[str]:
    merged = [label, *examples] if label.strip() else list(examples)
    return merged
