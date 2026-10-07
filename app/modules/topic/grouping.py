from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from app.config import Settings
from app.core.clustering import CosineThresholdClusterer, ClusterPoint, ClusterResult
from app.modules.topic.contracts import (
    TopicGroupMemberOut,
    TopicGroupOut,
    TopicUnassignedOut,
)
from app.modules.topic.scoring import cohesion_score, heuristic_confidence


def build_clusterer(settings: Settings) -> CosineThresholdClusterer:
    member_floor = min(
        settings.topic_min_member_similarity,
        settings.topic_assignment_min_score,
    )
    return CosineThresholdClusterer(
        similarity_threshold=settings.topic_cluster_similarity_threshold,
        min_member_similarity=member_floor,
        min_group_size=settings.topic_min_group_size,
    )


def cluster_points(points: Sequence[ClusterPoint], settings: Settings) -> ClusterResult:
    return build_clusterer(settings).cluster(points)


def materialize_groups_with_scores(
    *,
    cluster_result: ClusterResult,
    texts_by_ref: Mapping[str, str],
    score_to_rep: Mapping[str, Mapping[str, float]],
    settings: Settings,
) -> tuple[list[TopicGroupOut], list[TopicUnassignedOut], int]:
    """Map generic clusters → Topic proposal groups.

    suggested_label is the medoid member text (deterministic, no LLM).
    similarity_score is cosine-to-representative (not probability).
    confidence is a heuristic rescale above the assignment floor.
    """
    groups: list[TopicGroupOut] = []
    assigned: set[str] = set()
    low_confidence = 0

    for index, cluster in enumerate(cluster_result.groups, start=1):
        group_ref = f"g-{index:04d}"
        rep_ref = cluster.representative_ref
        member_scores = score_to_rep.get(cluster.group_key, {})
        members: list[TopicGroupMemberOut] = []
        scores: list[float] = []

        for ref in cluster.member_refs:
            if ref == rep_ref:
                sim = 1.0
            else:
                sim = float(
                    member_scores.get(ref, cluster.mean_score_to_representative)
                )
            conf = heuristic_confidence(sim, settings.topic_assignment_min_score)
            if conf < settings.topic_low_confidence_score and ref != rep_ref:
                low_confidence += 1
            members.append(
                TopicGroupMemberOut(
                    keyword_ref=ref,
                    text=texts_by_ref[ref],
                    similarity_score=round(sim, 6),
                    confidence=round(conf, 6),
                    is_representative=(ref == rep_ref),
                )
            )
            scores.append(sim)
            assigned.add(ref)

        members.sort(key=lambda m: (not m.is_representative, m.keyword_ref))
        mean_sim = float(sum(scores) / len(scores)) if scores else 0.0
        min_sim = float(min(scores)) if scores else 0.0
        groups.append(
            TopicGroupOut(
                group_ref=group_ref,
                suggested_label=texts_by_ref[rep_ref],
                member_count=len(members),
                mean_similarity=round(mean_sim, 6),
                min_similarity=round(min_sim, 6),
                cohesion=round(cohesion_score(mean_sim, min_sim), 6),
                members=members,
            )
        )

    unassigned: list[TopicUnassignedOut] = []
    for ref in cluster_result.unassigned_refs:
        unassigned.append(
            TopicUnassignedOut(
                keyword_ref=ref,
                text=texts_by_ref[ref],
                reason="below_threshold_or_small_component",
            )
        )
    for ref, text in texts_by_ref.items():
        if ref in assigned or any(u.keyword_ref == ref for u in unassigned):
            continue
        unassigned.append(TopicUnassignedOut(keyword_ref=ref, text=text, reason="not_grouped"))

    unassigned.sort(key=lambda row: row.keyword_ref)
    groups.sort(key=lambda g: g.group_ref)
    return groups, unassigned, low_confidence


def group_size_histogram(groups: Sequence[TopicGroupOut]) -> dict[str, int]:
    counter: Counter[str] = Counter(str(g.member_count) for g in groups)
    return dict(sorted(counter.items(), key=lambda item: int(item[0])))
