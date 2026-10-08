from __future__ import annotations

from statistics import median

from app.modules.internal_link_v2.contracts import (
    DistributionMetricsOut,
    InternalLinkCandidateIn,
    InternalLinkRankRequest,
    InternalLinkRankResponse,
    InternalLinkSuggestionOut,
    RankComponentOut,
    RejectedCandidateOut,
)


def rank_internal_links(request: InternalLinkRankRequest) -> InternalLinkRankResponse:
    if request.candidate_boundary != "topic_group":
        raise ValueError("topic_group boundary is required; global site scan is refused")

    metrics = _metrics(request.candidates)
    scale = max(1.0, float(metrics.max_inbound))
    accepted: list[InternalLinkCandidateIn] = []
    rejected: list[RejectedCandidateOut] = []
    for candidate in request.candidates:
        reason = _guard_reason(request.source_ref, candidate)
        if reason is not None:
            rejected.append(RejectedCandidateOut(ref=candidate.ref, reason=reason))
            continue
        accepted.append(candidate)

    suggestions: list[InternalLinkSuggestionOut] = []
    selected: set[str] = set()
    pool = list(accepted)
    while pool and len(suggestions) < request.limit:
        best_index = 0
        best_components: RankComponentOut | None = None
        for index, candidate in enumerate(pool):
            components = _components(
                candidate,
                metrics.median_inbound,
                scale,
                candidate.ref in selected,
                request.policy.overuse_weight,
                request.policy.underlinked_bonus,
                request.policy.repetition_penalty,
            )
            if best_components is None or components.final_score > best_components.final_score:
                best_index = index
                best_components = components
        assert best_components is not None
        chosen = pool.pop(best_index)
        selected.add(chosen.ref)
        suggestions.append(
            InternalLinkSuggestionOut(
                ref=chosen.ref,
                topic_group_ref=chosen.topic_group_ref,
                url=chosen.url,
                score=best_components.final_score,
                components=best_components,
            )
        )

    return InternalLinkRankResponse(
        source_ref=request.source_ref,
        candidate_boundary="topic_group",
        suggestions=suggestions,
        rejected=rejected,
        metrics=metrics,
    )


def _guard_reason(source_ref: str, candidate: InternalLinkCandidateIn) -> str | None:
    if candidate.same_as_source or candidate.ref == source_ref:
        return "self_link"
    if not candidate.eligible:
        return "ineligible"
    url = candidate.url.strip()
    if url == "" or not (url.startswith("https://") or url.startswith("http://") or url.startswith("/")):
        return "invalid_url"
    if candidate.already_linked_from_source:
        return "duplicate_source_target"
    return None


def _components(
    candidate: InternalLinkCandidateIn,
    median_inbound: float,
    scale: float,
    already_selected: bool,
    overuse_weight: float,
    underlinked_bonus: float,
    repetition_penalty: float,
) -> RankComponentOut:
    overuse = max(0.0, float(candidate.inbound_count) - median_inbound) / scale
    overuse_penalty = overuse_weight * overuse
    bonus = underlinked_bonus if candidate.inbound_count == 0 else 0.0
    repetition = repetition_penalty if already_selected else 0.0
    # Relevance stays primary: distribution can nudge close scores, not invert a clear topical gap.
    adjustment = max(-0.08, min(0.08, bonus - overuse_penalty - repetition))
    final_score = candidate.relevance + adjustment
    return RankComponentOut(
        relevance=candidate.relevance,
        overuse_penalty=overuse_penalty,
        underlinked_bonus=bonus,
        repetition_penalty=repetition,
        final_score=final_score,
    )


def _metrics(candidates: list[InternalLinkCandidateIn]) -> DistributionMetricsOut:
    inbound = [item.inbound_count for item in candidates]
    outbound = {item.ref: item.outbound_count for item in candidates}
    inbound_map = {item.ref: item.inbound_count for item in candidates}
    if not inbound:
        return DistributionMetricsOut(
            articles_with_zero_inbound=0,
            inbound_counts={},
            median_inbound=0.0,
            p90_inbound=0.0,
            max_inbound=0,
            outbound_counts={},
            repeated_target_refs=[],
            top_link_concentration=0.0,
        )
    ordered = sorted(inbound)
    index = min(len(ordered) - 1, max(0, int(round(0.9 * (len(ordered) - 1)))))
    total = sum(inbound)
    top_n = max(1, len(ordered) // 5 or 1)
    top_sum = sum(sorted(inbound, reverse=True)[:top_n])
    counts: dict[str, int] = {}
    for item in candidates:
        counts[item.ref] = counts.get(item.ref, 0) + 1
    repeated = sorted(ref for ref, count in counts.items() if count > 1)
    return DistributionMetricsOut(
        articles_with_zero_inbound=sum(1 for count in inbound if count == 0),
        inbound_counts=inbound_map,
        median_inbound=float(median(inbound)),
        p90_inbound=float(ordered[index]),
        max_inbound=max(inbound),
        outbound_counts=outbound,
        repeated_target_refs=repeated,
        top_link_concentration=(top_sum / total) if total else 0.0,
    )
