from __future__ import annotations

import numpy as np

from app.core.clustering import ClusterPoint, CosineThresholdClusterer


def _vec(*values: float) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr / np.linalg.norm(arr)
    return tuple(float(x) for x in arr)


def test_similar_points_group_and_outlier_stays_unassigned() -> None:
    clusterer = CosineThresholdClusterer(
        similarity_threshold=0.85,
        min_member_similarity=0.80,
        min_group_size=2,
    )
    points = [
        ClusterPoint("a", _vec(1, 0, 0)),
        ClusterPoint("b", _vec(0.98, 0.2, 0)),
        ClusterPoint("c", _vec(0, 1, 0)),
    ]
    result = clusterer.cluster(points)
    assert len(result.groups) == 1
    assert set(result.groups[0].member_refs) == {"a", "b"}
    assert result.groups[0].representative_ref in {"a", "b"}
    assert result.unassigned_refs == ("c",)


def test_deterministic_for_identical_input() -> None:
    clusterer = CosineThresholdClusterer(similarity_threshold=0.7, min_member_similarity=0.6)
    points = [
        ClusterPoint("z", _vec(1, 0.1, 0)),
        ClusterPoint("a", _vec(1, 0, 0)),
        ClusterPoint("m", _vec(0.95, 0.2, 0)),
    ]
    first = clusterer.cluster(points)
    second = clusterer.cluster(list(reversed(points)))
    assert first.groups[0].member_refs == second.groups[0].member_refs
    assert first.groups[0].representative_ref == second.groups[0].representative_ref
    assert first.unassigned_refs == second.unassigned_refs
