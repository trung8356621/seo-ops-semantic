from __future__ import annotations

from typing import Mapping, Sequence

from app.config import Settings
from app.core.clustering import (
    ClusterPoint,
    ClusterResult,
    CosineAverageLinkageClusterer,
    CosineThresholdClusterer,
    CosineThresholdGreedyMedoidV2,
)
from app.modules.keyword_grouping.contracts import GroupMemberOut, GroupOut, UnassignedOut
from app.modules.keyword_grouping.scoring import cohesion_score, heuristic_confidence


def build_clusterer(settings: Settings):
    algorithm = (settings.topic_cluster_algorithm or "average_linkage").strip().lower()
    if algorithm in {"average_linkage", "cosine_average_linkage_v1"}:
        return CosineAverageLinkageClusterer(
            similarity_threshold=settings.topic_cluster_similarity_threshold,
            min_group_size=settings.topic_min_group_size,
        )
    if algorithm in {"greedy_medoid_v2", "cosine_threshold_greedy_medoid_v2"}:
        return CosineThresholdGreedyMedoidV2(
            seed_density_threshold=settings.topic_cluster_similarity_threshold,
            member_similarity_threshold=settings.topic_min_member_similarity,
            min_group_size=settings.topic_min_group_size,
        )
    if algorithm in {"greedy_medoid_v1", "cosine_threshold_greedy_medoid_v1", "legacy"}:
        return CosineThresholdClusterer(
            similarity_threshold=settings.topic_cluster_similarity_threshold,
            min_member_similarity=settings.topic_min_member_similarity,
            min_group_size=settings.topic_min_group_size,
        )
    raise ValueError(f"unsupported TOPIC_CLUSTER_ALGORITHM={settings.topic_cluster_algorithm!r}")


def cluster_points(points: Sequence[ClusterPoint], settings: Settings) -> ClusterResult:
    return build_clusterer(settings).cluster(points)


def materialize_groups_with_scores(
    *,
    cluster_result: ClusterResult,
    texts_by_ref: Mapping[str, str],
    score_to_rep: Mapping[str, Mapping[str, float]],
    settings: Settings,
) -> tuple[list[GroupOut], list[UnassignedOut], int]:
    """Map generic clusters → keyword groups (representative = medoid only).

    Assignment guard: members with similarity_to_representative <
    TOPIC_ASSIGNMENT_MIN_SCORE are moved to unassigned (except the representative).

    representative_text is the medoid member text (deterministic, no LLM).
    similarity_score is cosine-to-representative (not probability).
    """
    groups: list[GroupOut] = []
    assigned: set[str] = set()
    low_confidence = 0
    assignment_rejects: list[UnassignedOut] = []

    for index, cluster in enumerate(cluster_result.groups, start=1):
        group_ref = f"g-{index:04d}"
        rep_ref = cluster.representative_ref
        member_scores = score_to_rep.get(cluster.group_key, {})
        kept: list[GroupMemberOut] = []
        scores: list[float] = []

        for ref in cluster.member_refs:
            if ref == rep_ref:
                sim = 1.0
            else:
                sim = float(
                    member_scores.get(ref, cluster.mean_score_to_representative)
                )
            if ref != rep_ref and sim < settings.topic_assignment_min_score:
                assignment_rejects.append(
                    UnassignedOut(
                        ref=ref,
                        text=texts_by_ref[ref],
                        reason="below_assignment_min_score",
                    )
                )
                continue

            conf = heuristic_confidence(sim, settings.topic_assignment_min_score)
            if conf < settings.topic_low_confidence_score and ref != rep_ref:
                low_confidence += 1
            kept.append(
                GroupMemberOut(
                    ref=ref,
                    text=texts_by_ref[ref],
                    similarity_score=round(sim, 6),
                    is_representative=(ref == rep_ref),
                )
            )
            scores.append(sim)
            assigned.add(ref)

        if len(kept) < settings.topic_min_group_size:
            for member in kept:
                if member.ref in assigned:
                    assigned.discard(member.ref)
                assignment_rejects.append(
                    UnassignedOut(
                        ref=member.ref,
                        text=member.text,
                        reason="below_min_group_size_after_assignment",
                    )
                )
            continue

        kept.sort(key=lambda m: (not m.is_representative, m.ref))
        mean_sim = float(sum(scores) / len(scores)) if scores else 0.0
        min_sim = float(min(scores)) if scores else 0.0
        groups.append(
            GroupOut(
                group_ref=group_ref,
                representative_ref=rep_ref,
                representative_text=texts_by_ref[rep_ref],
                member_count=len(kept),
                mean_similarity=round(mean_sim, 6),
                min_similarity=round(min_sim, 6),
                cohesion=round(cohesion_score(mean_sim, min_sim), 6),
                members=kept,
            )
        )

    unassigned: list[UnassignedOut] = []
    seen_unassigned: set[str] = set()
    for row in assignment_rejects:
        if row.ref in seen_unassigned or row.ref in assigned:
            continue
        unassigned.append(row)
        seen_unassigned.add(row.ref)

    for ref in cluster_result.unassigned_refs:
        if ref in assigned or ref in seen_unassigned:
            continue
        unassigned.append(
            UnassignedOut(
                ref=ref,
                text=texts_by_ref[ref],
                reason="below_threshold_or_small_component",
            )
        )
        seen_unassigned.add(ref)

    for ref, text in texts_by_ref.items():
        if ref in assigned or ref in seen_unassigned:
            continue
        unassigned.append(UnassignedOut(ref=ref, text=text, reason="not_grouped"))
        seen_unassigned.add(ref)

    unassigned.sort(key=lambda row: row.ref)
    groups.sort(key=lambda g: g.group_ref)
    return groups, unassigned, low_confidence
