from __future__ import annotations

from typing import Sequence

import numpy as np

from app.core.clustering.contracts import (
    ClusterGroup,
    ClusterPoint,
    ClusterResult,
)

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
except ImportError:  # pragma: no cover - dependency declared in pyproject
    fcluster = None  # type: ignore[assignment]
    linkage = None  # type: ignore[assignment]
    squareform = None  # type: ignore[assignment]


class CosineAverageLinkageClusterer:
    """Deterministic average-linkage agglomerative clustering on cosine distance.

    - Distance = 1 - cosine_similarity (L2-normalized vectors).
    - Clusters are cut where average inter-cluster distance exceeds
      ``1 - similarity_threshold``.
    - Groups smaller than ``min_group_size`` become unassigned.
    - Representative = deterministic medoid (max mean in-group similarity).
    - No Topic/Keyword/Laravel knowledge.
    """

    ALGORITHM = "cosine_average_linkage_v1"

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.74,
        min_group_size: int = 2,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if min_group_size < 1:
            raise ValueError("min_group_size must be >= 1")
        if linkage is None or fcluster is None or squareform is None:
            raise RuntimeError("scipy is required for CosineAverageLinkageClusterer")
        self.similarity_threshold = float(similarity_threshold)
        self.min_group_size = int(min_group_size)
        self.distance_threshold = 1.0 - self.similarity_threshold

    def cluster(self, points: Sequence[ClusterPoint]) -> ClusterResult:
        if not points:
            return ClusterResult(
                groups=(),
                unassigned_refs=(),
                algorithm=self.ALGORITHM,
                config=self._config(),
                diagnostics={"point_count": 0},
            )

        ordered = sorted(points, key=lambda p: p.ref)
        refs = [p.ref for p in ordered]
        matrix = self._normalized_matrix(ordered)
        similarity = matrix @ matrix.T
        n = len(refs)

        if n == 1:
            return ClusterResult(
                groups=(),
                unassigned_refs=(refs[0],),
                algorithm=self.ALGORITHM,
                config=self._config(),
                diagnostics={"point_count": 1, "group_count": 0, "unassigned_count": 1},
            )

        # Numerical safety: clamp cosine into [-1, 1] before distance.
        similarity = np.clip(similarity, -1.0, 1.0)
        distance = np.clip(1.0 - similarity, 0.0, 2.0)
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        tree = linkage(condensed, method="average")
        labels = fcluster(tree, t=self.distance_threshold, criterion="distance")

        buckets: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            buckets.setdefault(int(label), []).append(idx)

        groups: list[ClusterGroup] = []
        unassigned: list[str] = []
        group_index = 0
        for label in sorted(buckets.keys()):
            members = sorted(buckets[label], key=lambda i: refs[i])
            if len(members) < self.min_group_size:
                for idx in members:
                    unassigned.append(refs[idx])
                continue
            medoid = self._medoid(members, similarity)
            scores = [float(similarity[i, medoid]) for i in members]
            group_index += 1
            groups.append(
                ClusterGroup(
                    group_key=f"c-{group_index:04d}",
                    member_refs=tuple(refs[i] for i in members),
                    representative_ref=refs[medoid],
                    mean_score_to_representative=float(sum(scores) / len(scores)),
                    min_score_to_representative=float(min(scores)),
                    metadata={"member_count": len(members), "linkage_label": label},
                )
            )

        unassigned_unique = tuple(sorted(set(unassigned)))
        return ClusterResult(
            groups=tuple(groups),
            unassigned_refs=unassigned_unique,
            algorithm=self.ALGORITHM,
            config=self._config(),
            diagnostics={
                "point_count": n,
                "group_count": len(groups),
                "unassigned_count": len(unassigned_unique),
                "distance_threshold": self.distance_threshold,
            },
        )

    def _config(self) -> dict[str, float | int | str]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "distance_threshold": self.distance_threshold,
            "min_group_size": self.min_group_size,
            "strategy": "average_linkage",
            "representative": "medoid",
            "metric": "cosine_distance",
        }

    @staticmethod
    def _normalized_matrix(points: Sequence[ClusterPoint]) -> np.ndarray:
        rows = []
        expected_dim: int | None = None
        for point in points:
            vector = np.asarray(point.vector, dtype=np.float64)
            if vector.ndim != 1 or vector.size == 0:
                raise ValueError(f"invalid vector for ref={point.ref}")
            if expected_dim is None:
                expected_dim = int(vector.size)
            elif int(vector.size) != expected_dim:
                raise ValueError("all vectors must share the same dimensions")
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"non-finite vector for ref={point.ref}")
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise ValueError(f"zero vector for ref={point.ref}")
            rows.append(vector / norm)
        return np.vstack(rows)

    @staticmethod
    def _medoid(indices: Sequence[int], similarity: np.ndarray) -> int:
        best_idx = indices[0]
        best_score = float("-inf")
        for candidate in indices:
            mean_score = float(np.mean(similarity[candidate, list(indices)]))
            if mean_score > best_score or (mean_score == best_score and candidate < best_idx):
                best_score = mean_score
                best_idx = candidate
        return best_idx
