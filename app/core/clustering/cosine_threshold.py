from __future__ import annotations

from typing import Sequence

import numpy as np

from app.core.clustering.contracts import (
    ClusterGroup,
    ClusterPoint,
    ClusterResult,
)


class CosineThresholdClusterer:
    """Deterministic cosine-threshold clustering via greedy medoid seeding.

    Algorithm:
    1. L2-normalize vectors and compute full similarity matrix.
    2. While points remain:
       - Choose the remaining seed with the most neighbors at
         ``similarity_threshold`` (tie-break: lower ref).
       - Form a candidate group = seed + neighbors with
         similarity(seed) >= ``min_member_similarity``.
       - If size < ``min_group_size``, mark the seed unassigned.
       - Else keep the group; representative = seed (medoid of the take).
    3. Leftovers are unassigned.

    Avoids single-linkage chaining. No Topic/Keyword/Laravel knowledge.
    Suitable for ~100–1500 points (O(n²) similarity).
    """

    ALGORITHM = "cosine_threshold_greedy_medoid_v1"

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.70,
        min_member_similarity: float = 0.62,
        min_group_size: int = 2,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if not 0.0 <= min_member_similarity <= 1.0:
            raise ValueError("min_member_similarity must be in [0, 1]")
        if min_group_size < 1:
            raise ValueError("min_group_size must be >= 1")
        self.similarity_threshold = float(similarity_threshold)
        self.min_member_similarity = float(min_member_similarity)
        self.min_group_size = int(min_group_size)

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

        remaining: set[int] = set(range(n))
        groups: list[ClusterGroup] = []
        unassigned: list[str] = []
        group_index = 0
        rejected_small_seeds = 0

        while remaining:
            seed = self._best_seed(remaining, similarity)
            neighbors = [
                idx
                for idx in remaining
                if idx != seed and float(similarity[seed, idx]) >= self.similarity_threshold
            ]
            candidate = [seed]
            for idx in sorted(neighbors, key=lambda i: (-float(similarity[seed, i]), refs[i])):
                if float(similarity[seed, idx]) >= self.min_member_similarity:
                    candidate.append(idx)

            if len(candidate) < self.min_group_size:
                unassigned.append(refs[seed])
                remaining.remove(seed)
                rejected_small_seeds += 1
                continue

            # Re-pick medoid among candidates for a stable representative.
            medoid = self._medoid(candidate, similarity)
            member_indices = sorted(candidate, key=lambda i: refs[i])
            scores = [float(similarity[i, medoid]) for i in member_indices]
            group_index += 1
            groups.append(
                ClusterGroup(
                    group_key=f"c-{group_index:04d}",
                    member_refs=tuple(refs[i] for i in member_indices),
                    representative_ref=refs[medoid],
                    mean_score_to_representative=float(sum(scores) / len(scores)),
                    min_score_to_representative=float(min(scores)),
                    metadata={"member_count": len(member_indices)},
                )
            )
            for idx in candidate:
                remaining.discard(idx)

        for idx in sorted(remaining, key=lambda i: refs[i]):
            unassigned.append(refs[idx])

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
                "rejected_small_seeds": rejected_small_seeds,
            },
        )

    def _config(self) -> dict[str, float | int | str]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "min_member_similarity": self.min_member_similarity,
            "min_group_size": self.min_group_size,
            "strategy": "greedy_medoid_seed",
            "representative": "medoid",
        }

    def _best_seed(self, remaining: set[int], similarity: np.ndarray) -> int:
        best = min(remaining)
        best_count = -1
        for idx in sorted(remaining):
            count = sum(
                1
                for other in remaining
                if other != idx and float(similarity[idx, other]) >= self.similarity_threshold
            )
            if count > best_count or (count == best_count and idx < best):
                best = idx
                best_count = count
        return best

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
            mean_score = float(np.mean([similarity[candidate, other] for other in indices]))
            if mean_score > best_score or (mean_score == best_score and candidate < best_idx):
                best_score = mean_score
                best_idx = candidate
        return best_idx
