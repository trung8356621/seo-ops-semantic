from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from app.config import Settings
from app.core.clustering import ClusterResult
from app.modules.keyword_grouping.grouping import build_clusterer, cluster_points
from app.modules.keyword_grouping.grouping import (
    materialize_groups_with_scores as materialize_keyword_groups,
)
from app.modules.keyword_grouping.scoring import heuristic_confidence
from app.modules.topic.contracts import (
    TopicGroupMemberOut,
    TopicGroupOut,
    TopicUnassignedOut,
)


def materialize_groups_with_scores(
    *,
    cluster_result: ClusterResult,
    texts_by_ref: Mapping[str, str],
    score_to_rep: Mapping[str, Mapping[str, float]],
    settings: Settings,
) -> tuple[list[TopicGroupOut], list[TopicUnassignedOut], int]:
    """Compatibility adapter: keyword groups → Topic proposal shape.

    suggested_label mirrors representative_text (medoid). Clustering core lives
    in ``app.modules.keyword_grouping``.
    """
    groups_raw, unassigned_raw, low_confidence = materialize_keyword_groups(
        cluster_result=cluster_result,
        texts_by_ref=texts_by_ref,
        score_to_rep=score_to_rep,
        settings=settings,
    )

    groups: list[TopicGroupOut] = []
    for group in groups_raw:
        members = [
            TopicGroupMemberOut(
                keyword_ref=m.ref,
                text=m.text,
                similarity_score=m.similarity_score,
                confidence=round(
                    heuristic_confidence(m.similarity_score, settings.topic_assignment_min_score),
                    6,
                ),
                is_representative=m.is_representative,
            )
            for m in group.members
        ]
        groups.append(
            TopicGroupOut(
                group_ref=group.group_ref,
                suggested_label=group.representative_text,
                member_count=group.member_count,
                mean_similarity=group.mean_similarity,
                min_similarity=group.min_similarity,
                cohesion=group.cohesion,
                members=members,
            )
        )

    unassigned = [
        TopicUnassignedOut(
            keyword_ref=row.ref,
            text=row.text,
            reason=row.reason,
        )
        for row in unassigned_raw
    ]
    return groups, unassigned, low_confidence


def group_size_histogram(groups: Sequence[TopicGroupOut]) -> dict[str, int]:
    counter: Counter[str] = Counter(str(g.member_count) for g in groups)
    return dict(sorted(counter.items(), key=lambda item: int(item[0])))


__all__ = [
    "build_clusterer",
    "cluster_points",
    "materialize_groups_with_scores",
    "group_size_histogram",
]
