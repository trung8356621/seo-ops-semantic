from __future__ import annotations

import numpy as np

from app.core.clustering import ClusterPoint, CosineAverageLinkageClusterer


def _vec(*values: float) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr / np.linalg.norm(arr)
    return tuple(float(x) for x in arr)


def test_average_linkage_keeps_families_separate() -> None:
    """Two tight families + one hub-like midpoint should not merge all."""
    clusterer = CosineAverageLinkageClusterer(similarity_threshold=0.85, min_group_size=2)
    points = [
        ClusterPoint("a1", _vec(1, 0, 0)),
        ClusterPoint("a2", _vec(0.98, 0.1, 0)),
        ClusterPoint("a3", _vec(0.97, 0.12, 0)),
        ClusterPoint("b1", _vec(0, 1, 0)),
        ClusterPoint("b2", _vec(0.1, 0.98, 0)),
        ClusterPoint("b3", _vec(0.12, 0.97, 0)),
        ClusterPoint("outlier", _vec(0, 0, 1)),
    ]
    result = clusterer.cluster(points)
    assert result.algorithm == "cosine_average_linkage_v1"
    member_sets = [set(g.member_refs) for g in result.groups]
    assert {"a1", "a2", "a3"} in member_sets or {"a1", "a2"} in member_sets or any(
        {"a1", "a2", "a3"}.issubset(s) for s in member_sets
    )
    assert any({"b1", "b2"}.issubset(s) for s in member_sets)
    assert "outlier" in result.unassigned_refs or all("outlier" not in s for s in member_sets)
    # Families must not collapse into one mega-group.
    assert max(len(s) for s in member_sets) < 6


def test_average_linkage_deterministic() -> None:
    clusterer = CosineAverageLinkageClusterer(similarity_threshold=0.8, min_group_size=2)
    points = [
        ClusterPoint("z", _vec(1, 0.05, 0)),
        ClusterPoint("a", _vec(1, 0, 0)),
        ClusterPoint("m", _vec(0.96, 0.2, 0)),
        ClusterPoint("q", _vec(0, 1, 0)),
    ]
    first = clusterer.cluster(points)
    second = clusterer.cluster(list(reversed(points)))
    assert [g.member_refs for g in first.groups] == [g.member_refs for g in second.groups]
    assert first.unassigned_refs == second.unassigned_refs
