from __future__ import annotations

import numpy as np

from app.core.clustering import ClusterPoint, CosineThresholdGreedyMedoidV2


def _vec(*values: float) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr / np.linalg.norm(arr)
    return tuple(float(x) for x in arr)


def test_member_threshold_stricter_than_density_excludes_weak_neighbors() -> None:
    """Density can see a point at 0.72; member bar 0.80 must reject it."""
    # Construct vectors with controlled cosine to seed (1,0,0).
    seed = _vec(1, 0, 0)
    strong = _vec(0.95, 0.3122, 0)  # ~0.95
    weak = _vec(0.72, 0.694, 0)  # ~0.72
    other_family = _vec(0, 1, 0)

    # Verify angles
    assert np.dot(seed, strong) > 0.90
    assert 0.70 < np.dot(seed, weak) < 0.80

    loose = CosineThresholdGreedyMedoidV2(
        seed_density_threshold=0.70,
        member_similarity_threshold=0.71,
        min_group_size=2,
    )
    loose_result = loose.cluster(
        [
            ClusterPoint("seed", seed),
            ClusterPoint("strong", strong),
            ClusterPoint("weak", weak),
            ClusterPoint("other", other_family),
        ]
    )
    loose_members = set(loose_result.groups[0].member_refs)
    assert "weak" in loose_members

    strict = CosineThresholdGreedyMedoidV2(
        seed_density_threshold=0.70,
        member_similarity_threshold=0.80,
        min_group_size=2,
    )
    strict_result = strict.cluster(
        [
            ClusterPoint("seed", seed),
            ClusterPoint("strong", strong),
            ClusterPoint("weak", weak),
            ClusterPoint("other", other_family),
        ]
    )
    strict_members = set(strict_result.groups[0].member_refs)
    assert "strong" in strict_members
    assert "weak" not in strict_members
    assert "weak" in strict_result.unassigned_refs or "weak" not in strict_members


def test_greedy_v2_algorithm_name() -> None:
    clusterer = CosineThresholdGreedyMedoidV2()
    assert clusterer.ALGORITHM == "cosine_threshold_greedy_medoid_v2"
