from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from app.config import Settings
from app.core.clustering import ClusterPoint
from app.core.text.lexical import pair_evidence
from app.modules.keyword_grouping.contracts import GroupMemberOut, GroupOut, UnassignedOut
from app.modules.keyword_grouping.scoring import cohesion_score

ALGORITHM = "hybrid_semantic_lexical_v1"


@dataclass
class HybridDiagnostics:
    lexical_reject_count: int = 0
    rescue_assignment_count: int = 0
    ambiguous_count: int = 0
    strategy: str = ALGORITHM
    semantic_candidate_edges: int = 0
    primary_group_count: int = 0


@dataclass
class _MutableGroup:
    members: set[str] = field(default_factory=set)
    representative_ref: str = ""


def _cosine_matrix(points: Sequence[ClusterPoint]) -> tuple[list[str], np.ndarray]:
    ordered = sorted(points, key=lambda p: p.ref)
    refs = [p.ref for p in ordered]
    matrix = np.asarray([p.vector for p in ordered], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    unit = matrix / norms
    sim = np.clip(unit @ unit.T, -1.0, 1.0)
    return refs, sim


def _sim(sim: np.ndarray, index: dict[str, int], a: str, b: str) -> float:
    return float(sim[index[a], index[b]])


def _is_bridge_node(
    ref: str,
    neighbors: Mapping[str, set[str]],
    compatible: Mapping[tuple[str, str], bool],
) -> bool:
    """Node adjacent to two mutually incompatible neighbors → bridging risk."""
    neigh = sorted(neighbors.get(ref, ()))
    for i, a in enumerate(neigh):
        for b in neigh[i + 1 :]:
            key = (a, b) if a < b else (b, a)
            if not compatible.get(key, False):
                return True
    return False


def _pick_medoid(members: set[str], sim: np.ndarray, index: dict[str, int]) -> str:
    best_ref = min(members)
    best_score = -1.0
    for cand in sorted(members):
        scores = [_sim(sim, index, cand, other) for other in members if other != cand]
        mean = float(sum(scores) / len(scores)) if scores else 1.0
        if mean > best_score or (mean == best_score and cand < best_ref):
            best_score = mean
            best_ref = cand
    return best_ref


def _compatible_with_all(
    ref: str,
    members: set[str],
    compatible: Mapping[tuple[str, str], bool],
) -> bool:
    for other in members:
        if other == ref:
            continue
        key = (ref, other) if ref < other else (other, ref)
        if not compatible.get(key, False):
            return False
    return True


def run_hybrid_semantic_lexical_v1(
    *,
    points: Sequence[ClusterPoint],
    texts_by_ref: Mapping[str, str],
    settings: Settings,
) -> tuple[list[GroupOut], list[UnassignedOut], HybridDiagnostics, dict[str, object]]:
    """Hybrid grouping: semantic candidates + lexical guard + rescue.

    Avoids transitive bridging by excluding ambiguous multi-family nodes from
    primary clique growth, then leaving them unassigned when rescue is ambiguous.
    """
    diag = HybridDiagnostics()
    config = {
        "semantic_candidate_floor": settings.keyword_group_semantic_floor,
        "rescue_semantic_floor": settings.keyword_group_rescue_semantic_floor,
        "containment_min": settings.keyword_group_containment_min,
        "min_group_size": settings.keyword_group_min_group_size,
        "ambiguity_margin": settings.keyword_group_ambiguity_margin,
    }

    if not points:
        return [], [], diag, config

    refs, sim = _cosine_matrix(points)
    index = {ref: i for i, ref in enumerate(refs)}
    floor = float(settings.keyword_group_semantic_floor)
    containment_min = float(settings.keyword_group_containment_min)
    min_size = int(settings.keyword_group_min_group_size)

    compatible: dict[tuple[str, str], bool] = {}
    conflict_pairs: set[tuple[str, str]] = set()
    edge_score: dict[tuple[str, str], float] = {}
    neighbors: dict[str, set[str]] = {ref: set() for ref in refs}

    for i, a in enumerate(refs):
        for b in refs[i + 1 :]:
            cosine = float(sim[i, index[b]])
            evidence = pair_evidence(
                texts_by_ref[a],
                texts_by_ref[b],
                containment_min=containment_min,
            )
            key = (a, b)
            if evidence.conflict or not evidence.compatible:
                compatible[key] = False
                if evidence.conflict and cosine >= floor:
                    conflict_pairs.add(key)
                    diag.lexical_reject_count += 1
                continue
            if cosine < floor:
                compatible[key] = False
                continue
            compatible[key] = True
            strength = cosine + (0.15 if evidence.shared_ngrams else 0.0) + 0.05 * evidence.containment
            edge_score[key] = strength
            neighbors[a].add(b)
            neighbors[b].add(a)
            diag.semantic_candidate_edges += 1

    bridges = {ref for ref in refs if _is_bridge_node(ref, neighbors, compatible)}
    primary_nodes = [ref for ref in refs if ref not in bridges]
    unassigned_reason: dict[str, str] = {}
    for ref in bridges:
        unassigned_reason[ref] = "ambiguous_multiple_groups"
        diag.ambiguous_count += 1

    # Greedy clique growth on non-bridge nodes, strongest edges first.
    assigned: set[str] = set()
    groups: list[_MutableGroup] = []
    ranked_edges = sorted(
        (
            (score, a, b)
            for (a, b), score in edge_score.items()
            if a in primary_nodes and b in primary_nodes
        ),
        key=lambda row: (-row[0], row[1], row[2]),
    )

    for _score, a, b in ranked_edges:
        if a in assigned and b in assigned:
            continue
        if a not in assigned and b not in assigned:
            group = _MutableGroup(members={a, b})
            # Grow greedily: add nodes fully compatible with all members.
            grew = True
            while grew:
                grew = False
                for cand in primary_nodes:
                    if cand in group.members or cand in assigned:
                        continue
                    if _compatible_with_all(cand, group.members, compatible):
                        group.members.add(cand)
                        grew = True
            if len(group.members) >= min_size:
                for member in group.members:
                    assigned.add(member)
                group.representative_ref = _pick_medoid(group.members, sim, index)
                groups.append(group)
            continue

        # Attach singleton edge endpoint into existing group if fully compatible.
        if a in assigned and b not in assigned:
            host = next(g for g in groups if a in g.members)
            if _compatible_with_all(b, host.members, compatible):
                host.members.add(b)
                assigned.add(b)
                host.representative_ref = _pick_medoid(host.members, sim, index)
        elif b in assigned and a not in assigned:
            host = next(g for g in groups if b in g.members)
            if _compatible_with_all(a, host.members, compatible):
                host.members.add(a)
                assigned.add(a)
                host.representative_ref = _pick_medoid(host.members, sim, index)

    # Drop undersized groups.
    kept_groups: list[_MutableGroup] = []
    for group in groups:
        if len(group.members) < min_size:
            for member in group.members:
                assigned.discard(member)
                unassigned_reason.setdefault(member, "below_min_group_size")
            continue
        kept_groups.append(group)
    groups = kept_groups
    diag.primary_group_count = len(groups)

    # Rescue pass for remaining non-ambiguous unassigned.
    rescue_floor = float(settings.keyword_group_rescue_semantic_floor)
    ambiguity_margin = float(settings.keyword_group_ambiguity_margin)
    for ref in refs:
        if ref in assigned:
            continue
        if unassigned_reason.get(ref) == "ambiguous_multiple_groups":
            continue

        candidates: list[tuple[float, int, _MutableGroup]] = []
        for gi, group in enumerate(groups):
            rep = group.representative_ref
            cosine = _sim(sim, index, ref, rep)
            if cosine < rescue_floor:
                continue
            evidence = pair_evidence(
                texts_by_ref[ref],
                texts_by_ref[rep],
                containment_min=containment_min,
            )
            if evidence.conflict or not evidence.compatible:
                continue
            if not _compatible_with_all(ref, group.members, compatible):
                # Soften: require compatibility with representative + >= half members
                ok = 0
                for member in group.members:
                    key = (ref, member) if ref < member else (member, ref)
                    if compatible.get(key, False) or (
                        pair_evidence(
                            texts_by_ref[ref],
                            texts_by_ref[member],
                            containment_min=containment_min,
                        ).compatible
                        and _sim(sim, index, ref, member) >= rescue_floor
                    ):
                        ok += 1
                if ok < max(1, (len(group.members) + 1) // 2):
                    continue
            score = cosine
            if evidence.shared_ngrams:
                score += 0.1
            candidates.append((score, gi, group))

        if not candidates:
            # Classify reason
            best_cos = max((_sim(sim, index, ref, other) for other in refs if other != ref), default=0.0)
            if best_cos < rescue_floor:
                unassigned_reason.setdefault(ref, "below_semantic_floor")
            elif any(
                pair_evidence(
                    texts_by_ref[ref],
                    texts_by_ref[other],
                    containment_min=containment_min,
                ).conflict
                for other in refs
                if other != ref and _sim(sim, index, ref, other) >= rescue_floor
            ):
                unassigned_reason.setdefault(ref, "lexical_conflict")
            else:
                unassigned_reason.setdefault(ref, "no_compatible_group")
            continue

        candidates.sort(key=lambda row: (-row[0], row[1]))
        best_score, _best_gi, best_group = candidates[0]
        if len(candidates) > 1:
            second = candidates[1][0]
            if abs(best_score - second) <= ambiguity_margin:
                unassigned_reason[ref] = "ambiguous_multiple_groups"
                diag.ambiguous_count += 1
                continue

        best_group.members.add(ref)
        assigned.add(ref)
        best_group.representative_ref = _pick_medoid(best_group.members, sim, index)
        unassigned_reason.pop(ref, None)
        diag.rescue_assignment_count += 1

    # Re-check bridges against formed groups (still ambiguous if 2+ fits).
    for ref in list(bridges):
        if ref in assigned:
            continue
        fits = 0
        for group in groups:
            cosine = _sim(sim, index, ref, group.representative_ref)
            if cosine < rescue_floor:
                continue
            evidence = pair_evidence(
                texts_by_ref[ref],
                texts_by_ref[group.representative_ref],
                containment_min=containment_min,
            )
            if evidence.compatible and not evidence.conflict:
                fits += 1
        if fits >= 2:
            unassigned_reason[ref] = "ambiguous_multiple_groups"
        elif fits == 0:
            unassigned_reason.setdefault(ref, "no_compatible_group")
        else:
            # Exactly one group — still prefer unassigned for known bridges unless
            # clearly single-family; bridges stay unassigned by policy.
            unassigned_reason[ref] = "ambiguous_multiple_groups"

    # Materialize outputs.
    out_groups: list[GroupOut] = []
    for index_g, group in enumerate(sorted(groups, key=lambda g: min(g.members)), start=1):
        rep = group.representative_ref or _pick_medoid(group.members, sim, index)
        members_out: list[GroupMemberOut] = []
        scores: list[float] = []
        for member in sorted(group.members):
            score = 1.0 if member == rep else _sim(sim, index, member, rep)
            members_out.append(
                GroupMemberOut(
                    ref=member,
                    text=texts_by_ref[member],
                    similarity_score=round(float(score), 6),
                    is_representative=(member == rep),
                )
            )
            scores.append(float(score))
        members_out.sort(key=lambda m: (not m.is_representative, m.ref))
        mean_sim = float(sum(scores) / len(scores)) if scores else 0.0
        min_sim = float(min(scores)) if scores else 0.0
        out_groups.append(
            GroupOut(
                group_ref=f"g-{index_g:04d}",
                representative_ref=rep,
                representative_text=texts_by_ref[rep],
                member_count=len(members_out),
                mean_similarity=round(mean_sim, 6),
                min_similarity=round(min_sim, 6),
                cohesion=round(cohesion_score(mean_sim, min_sim), 6),
                members=members_out,
            )
        )

    assigned_final = {m.ref for g in out_groups for m in g.members}
    unassigned: list[UnassignedOut] = []
    for ref in refs:
        if ref in assigned_final:
            continue
        reason = unassigned_reason.get(ref, "no_compatible_group")
        unassigned.append(UnassignedOut(ref=ref, text=texts_by_ref[ref], reason=reason))
    unassigned.sort(key=lambda row: row.ref)
    out_groups.sort(key=lambda g: g.group_ref)
    return out_groups, unassigned, diag, config
