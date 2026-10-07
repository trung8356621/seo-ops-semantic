from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class ClusterPoint:
    """Generic vector point. Feature modules supply their own opaque refs."""

    ref: str
    vector: Sequence[float]


@dataclass(frozen=True, slots=True)
class ClusterMember:
    ref: str
    score_to_representative: float


@dataclass(frozen=True, slots=True)
class ClusterGroup:
    group_key: str
    member_refs: tuple[str, ...]
    representative_ref: str
    mean_score_to_representative: float
    min_score_to_representative: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClusterResult:
    groups: tuple[ClusterGroup, ...]
    unassigned_refs: tuple[str, ...]
    algorithm: str
    config: Mapping[str, Any]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class Clusterer(Protocol):
    def cluster(self, points: Sequence[ClusterPoint]) -> ClusterResult: ...
