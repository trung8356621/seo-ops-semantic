"""Weighted semantic routing.

Relevance is the max cosine against a group's examples, so extra examples
cannot inflate a candidate. Weight only ranks groups that already clear the
existing tool-intent relevance floor.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.modules.tool_intent.catalog import DEFAULT_MIN_MARGIN, DEFAULT_MIN_POSITIVE_SCORE


@dataclass(frozen=True)
class WeightedTarget:
    ref: str
    weight: float


@dataclass(frozen=True)
class WeightedGroup:
    id: str
    examples: list[str]
    targets: list[WeightedTarget]
    enabled: bool = True


@dataclass(frozen=True)
class WeightedCandidate:
    ref: str
    semantic_relevance: float
    weight: float
    score: float
    group_id: str
    example: str


def group_relevance(similarities: list[float]) -> float:
    if not similarities:
        return 0.0
    return max(similarities)


def rank_weighted(
    relevances: dict[str, float],
    best_examples: dict[str, str],
    groups: list[WeightedGroup],
    *,
    floor: float = DEFAULT_MIN_POSITIVE_SCORE,
    margin: float = DEFAULT_MIN_MARGIN,
) -> tuple[str, str | None, list[WeightedCandidate]]:
    enabled = [group for group in groups if group.enabled and group.targets and group.examples]
    if not enabled:
        return "none", None, []

    max_weight = max(target.weight for group in enabled for target in group.targets)
    if max_weight <= 0:
        return "none", None, []

    best_by_ref: dict[str, WeightedCandidate] = {}
    for group in enabled:
        relevance = relevances.get(group.id, 0.0)
        example = best_examples.get(group.id, group.examples[0])
        for target in group.targets:
            score = relevance * (target.weight / max_weight)
            current = best_by_ref.get(target.ref)
            if current is None or score > current.score:
                best_by_ref[target.ref] = WeightedCandidate(
                    ref=target.ref,
                    semantic_relevance=relevance,
                    weight=target.weight,
                    score=score,
                    group_id=group.id,
                    example=example,
                )

    ordered = sorted(best_by_ref.values(), key=lambda item: item.score, reverse=True)
    eligible = [item for item in ordered if item.semantic_relevance >= floor]
    if not eligible:
        return "none", None, ordered[:4]

    top = eligible[0]
    second = eligible[1] if len(eligible) > 1 else None
    if second is not None and (top.score - second.score) < margin:
        return "ambiguous", None, [top, second, *ordered[2:4]]
    rest = [item for item in ordered if item.ref != top.ref]
    return "confident", top.ref, [top, *rest[:3]]
