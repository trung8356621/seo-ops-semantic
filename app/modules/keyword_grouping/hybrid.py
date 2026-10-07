from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from app.config import Settings
from app.core.clustering import ClusterPoint
from app.core.text.lexical import content_tokens, informative_ngrams, pair_evidence
from app.modules.keyword_grouping.contracts import GroupMemberOut, GroupOut, UnassignedOut
from app.modules.keyword_grouping.scoring import cohesion_score

ALGORITHM = "hybrid_semantic_lexical_v1"


class PairRelation(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass
class HybridDiagnostics:
    lexical_reject_count: int = 0
    rescue_assignment_count: int = 0
    ambiguous_count: int = 0
    strategy: str = ALGORITHM
    semantic_candidate_edges: int = 0
    primary_group_count: int = 0


@dataclass
class GroupCompatResult:
    accepted: bool
    score: float
    conflict: bool
    shared_anchors: frozenset[str] = field(default_factory=frozenset)


@dataclass
class _MutableGroup:
    members: set[str] = field(default_factory=set)
    representative_ref: str = ""
    anchor_ngrams: frozenset[str] = field(default_factory=frozenset)


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


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def classify_pair_relation(
    *,
    text_a: str,
    text_b: str,
    cosine: float,
    semantic_floor: float,
    containment_min: float,
) -> tuple[PairRelation, object]:
    """Three-state pair relation. Cosine alone never yields COMPATIBLE."""
    evidence = pair_evidence(text_a, text_b, containment_min=containment_min)
    if evidence.conflict:
        return PairRelation.INCOMPATIBLE, evidence
    if evidence.compatible and cosine >= semantic_floor:
        return PairRelation.COMPATIBLE, evidence
    return PairRelation.UNKNOWN, evidence


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


def _member_ngrams(text: str) -> frozenset[str]:
    return informative_ngrams(content_tokens(text))


def compute_group_anchors(
    members: set[str],
    texts_by_ref: Mapping[str, str],
) -> frozenset[str]:
    """Discover repeated informative n-grams from members (not hardcoded)."""
    counts: Counter[str] = Counter()
    for ref in members:
        for ng in _member_ngrams(texts_by_ref[ref]):
            # Prefer multi-token anchors; length-2 carries most modifier signal.
            if len(ng.split()) >= 2:
                counts[ng] += 1
    if not counts:
        return frozenset()
    # Require support from >=2 members when group is large enough; else keep all.
    min_support = 2 if len(members) >= 2 else 1
    anchors = {ng for ng, c in counts.items() if c >= min_support}
    if anchors:
        return frozenset(anchors)
    # Fallback: most frequent bigrams.
    top = [ng for ng, _ in counts.most_common(3)]
    return frozenset(top)


def _conflict_ngram_pairs_from_incompatible(
    relations: Mapping[tuple[str, str], PairRelation],
    texts_by_ref: Mapping[str, str],
    containment_min: float,
) -> frozenset[tuple[str, str]]:
    """Discover exclusive informative-ngram pairs from INCOMPATIBLE edges."""
    pairs: set[tuple[str, str]] = set()
    for (a, b), rel in relations.items():
        if rel is not PairRelation.INCOMPATIBLE:
            continue
        evidence = pair_evidence(
            texts_by_ref[a],
            texts_by_ref[b],
            containment_min=containment_min,
        )
        # Prefer length-2 exclusives as competing modifiers.
        ex_a = {ng for ng in evidence.exclusive_ngrams_a if len(ng.split()) == 2}
        ex_b = {ng for ng in evidence.exclusive_ngrams_b if len(ng.split()) == 2}
        for left in ex_a:
            for right in ex_b:
                pairs.add(_pair_key(left, right))
    return frozenset(pairs)


def _is_dual_anchor_node(
    ref: str,
    texts_by_ref: Mapping[str, str],
    conflict_ngram_pairs: frozenset[tuple[str, str]],
) -> bool:
    """True when keyword contains both sides of a discovered conflict n-gram pair."""
    ngrams = _member_ngrams(texts_by_ref[ref])
    for left, right in conflict_ngram_pairs:
        if left in ngrams and right in ngrams:
            return True
    return False


def evaluate_keyword_against_group(
    *,
    ref: str,
    group: _MutableGroup,
    texts_by_ref: Mapping[str, str],
    sim: np.ndarray,
    index: dict[str, int],
    relations: Mapping[tuple[str, str], PairRelation],
    semantic_floor: float,
    containment_min: float,
) -> GroupCompatResult:
    """Group-level compatibility (not full pairwise clique)."""
    if not group.members:
        return GroupCompatResult(accepted=False, score=0.0, conflict=False)

    rep = group.representative_ref or next(iter(sorted(group.members)))
    cosine_rep = _sim(sim, index, ref, rep)
    if cosine_rep < semantic_floor:
        return GroupCompatResult(accepted=False, score=cosine_rep, conflict=False)

    anchors = group.anchor_ngrams or compute_group_anchors(group.members, texts_by_ref)
    cand_ngrams = _member_ngrams(texts_by_ref[ref])
    shared_anchors = frozenset(cand_ngrams & anchors)

    # Lexical vs representative.
    ev_rep = pair_evidence(
        texts_by_ref[ref],
        texts_by_ref[rep],
        containment_min=containment_min,
    )

    # Strong conflict: exclusive modifiers vs group and no shared group anchor.
    if ev_rep.conflict and not shared_anchors:
        # Confirm against a majority of members (one outlier must not veto).
        conflict_hits = 0
        checked = 0
        for member in group.members:
            checked += 1
            ev = pair_evidence(
                texts_by_ref[ref],
                texts_by_ref[member],
                containment_min=containment_min,
            )
            if ev.conflict and not (cand_ngrams & _member_ngrams(texts_by_ref[member])):
                conflict_hits += 1
        if checked > 0 and conflict_hits / checked >= 0.5:
            return GroupCompatResult(
                accepted=False,
                score=cosine_rep,
                conflict=True,
                shared_anchors=shared_anchors,
            )

    # Member support via positive relations or shared n-grams.
    support = 0
    for member in group.members:
        key = _pair_key(ref, member)
        if relations.get(key) is PairRelation.COMPATIBLE:
            support += 1
            continue
        ev = pair_evidence(
            texts_by_ref[ref],
            texts_by_ref[member],
            containment_min=containment_min,
        )
        if ev.compatible and not ev.conflict:
            # Lexical OK; allow slightly softer semantic vs individual members.
            if _sim(sim, index, ref, member) >= semantic_floor * 0.9:
                support += 1

    min_support = 1 if len(group.members) <= 2 else max(1, (len(group.members) + 2) // 3)
    has_rep_compat = (ev_rep.compatible and not ev_rep.conflict) or bool(shared_anchors)
    accepted = has_rep_compat and (bool(shared_anchors) or support >= min_support or ev_rep.compatible)

    score = cosine_rep
    if shared_anchors:
        score += 0.12 + 0.03 * min(3, len(shared_anchors))
    score += 0.02 * support

    return GroupCompatResult(
        accepted=accepted,
        score=float(score),
        conflict=False,
        shared_anchors=shared_anchors,
    )


def run_hybrid_semantic_lexical_v1(
    *,
    points: Sequence[ClusterPoint],
    texts_by_ref: Mapping[str, str],
    settings: Settings,
) -> tuple[list[GroupOut], list[UnassignedOut], HybridDiagnostics, dict[str, object]]:
    """Hybrid V2.1: positive-edge graph + group anchors + post-group ambiguity."""
    diag = HybridDiagnostics()
    config = {
        "semantic_candidate_floor": settings.keyword_group_semantic_floor,
        "rescue_semantic_floor": settings.keyword_group_rescue_semantic_floor,
        "containment_min": settings.keyword_group_containment_min,
        "min_group_size": settings.keyword_group_min_group_size,
        "ambiguity_margin": settings.keyword_group_ambiguity_margin,
        "version": "v2.1",
    }

    if not points:
        return [], [], diag, config

    refs, sim = _cosine_matrix(points)
    index = {ref: i for i, ref in enumerate(refs)}
    floor = float(settings.keyword_group_semantic_floor)
    rescue_floor = float(settings.keyword_group_rescue_semantic_floor)
    containment_min = float(settings.keyword_group_containment_min)
    min_size = int(settings.keyword_group_min_group_size)
    ambiguity_margin = float(settings.keyword_group_ambiguity_margin)

    relations: dict[tuple[str, str], PairRelation] = {}
    edge_score: dict[tuple[str, str], float] = {}
    neighbors: dict[str, set[str]] = {ref: set() for ref in refs}

    for i, a in enumerate(refs):
        for b in refs[i + 1 :]:
            cosine = float(sim[i, index[b]])
            rel, evidence = classify_pair_relation(
                text_a=texts_by_ref[a],
                text_b=texts_by_ref[b],
                cosine=cosine,
                semantic_floor=floor,
                containment_min=containment_min,
            )
            key = (a, b)
            relations[key] = rel
            if rel is PairRelation.INCOMPATIBLE and cosine >= floor:
                diag.lexical_reject_count += 1
                continue
            if rel is not PairRelation.COMPATIBLE:
                continue
            # Positive edge only.
            shared = getattr(evidence, "shared_ngrams", frozenset())
            containment = float(getattr(evidence, "containment", 0.0))
            strength = cosine + (0.15 if shared else 0.0) + 0.05 * containment
            edge_score[key] = strength
            neighbors[a].add(b)
            neighbors[b].add(a)
            diag.semantic_candidate_edges += 1

    conflict_ngram_pairs = _conflict_ngram_pairs_from_incompatible(
        relations, texts_by_ref, containment_min
    )
    dual_anchor = {
        ref for ref in refs if _is_dual_anchor_node(ref, texts_by_ref, conflict_ngram_pairs)
    }

    primary_pool = [ref for ref in refs if ref not in dual_anchor]
    unassigned_reason: dict[str, str] = {}
    assigned: set[str] = set()
    groups: list[_MutableGroup] = []

    ranked_edges = sorted(
        (
            (score, a, b)
            for (a, b), score in edge_score.items()
            if a in primary_pool and b in primary_pool
        ),
        key=lambda row: (-row[0], row[1], row[2]),
    )

    def _refresh(group: _MutableGroup) -> None:
        group.representative_ref = _pick_medoid(group.members, sim, index)
        group.anchor_ngrams = compute_group_anchors(group.members, texts_by_ref)

    def _try_join(ref: str, group: _MutableGroup, *, semantic_floor: float) -> bool:
        result = evaluate_keyword_against_group(
            ref=ref,
            group=group,
            texts_by_ref=texts_by_ref,
            sim=sim,
            index=index,
            relations=relations,
            semantic_floor=semantic_floor,
            containment_min=containment_min,
        )
        if not result.accepted or result.conflict:
            return False
        group.members.add(ref)
        assigned.add(ref)
        unassigned_reason.pop(ref, None)
        _refresh(group)
        return True

    for _score, a, b in ranked_edges:
        if a in assigned and b in assigned:
            continue
        if a not in assigned and b not in assigned:
            seed = _MutableGroup(members={a, b})
            _refresh(seed)
            grew = True
            while grew:
                grew = False
                for cand in primary_pool:
                    if cand in seed.members or cand in assigned:
                        continue
                    if _try_join(cand, seed, semantic_floor=floor):
                        grew = True
            if len(seed.members) >= min_size:
                for member in seed.members:
                    assigned.add(member)
                groups.append(seed)
            else:
                for member in list(seed.members):
                    assigned.discard(member)
            continue

        if a in assigned and b not in assigned:
            host = next(g for g in groups if a in g.members)
            _try_join(b, host, semantic_floor=floor)
        elif b in assigned and a not in assigned:
            host = next(g for g in groups if b in g.members)
            _try_join(a, host, semantic_floor=floor)

    kept: list[_MutableGroup] = []
    for group in groups:
        if len(group.members) < min_size:
            for member in group.members:
                assigned.discard(member)
                unassigned_reason.setdefault(member, "below_min_group_size")
            continue
        _refresh(group)
        kept.append(group)
    groups = kept
    diag.primary_group_count = len(groups)

    # Rescue / assignment pass — dual-anchor nodes included here only.
    pending = [ref for ref in refs if ref not in assigned]
    for ref in pending:
        candidates: list[tuple[float, int, _MutableGroup]] = []
        for gi, group in enumerate(groups):
            result = evaluate_keyword_against_group(
                ref=ref,
                group=group,
                texts_by_ref=texts_by_ref,
                sim=sim,
                index=index,
                relations=relations,
                semantic_floor=rescue_floor,
                containment_min=containment_min,
            )
            if result.conflict:
                continue
            if not result.accepted:
                continue
            candidates.append((result.score, gi, group))

        if not candidates:
            best_cos = max(
                (_sim(sim, index, ref, other) for other in refs if other != ref),
                default=0.0,
            )
            if best_cos < rescue_floor:
                unassigned_reason.setdefault(ref, "below_semantic_floor")
            else:
                # Only mark lexical_conflict if no group is lexically compatible.
                any_group_lex = False
                any_conflict_only = False
                for group in groups:
                    rep = group.representative_ref
                    ev = pair_evidence(
                        texts_by_ref[ref],
                        texts_by_ref[rep],
                        containment_min=containment_min,
                    )
                    anchors = group.anchor_ngrams
                    shared = _member_ngrams(texts_by_ref[ref]) & anchors
                    if (ev.compatible and not ev.conflict) or shared:
                        any_group_lex = True
                    if ev.conflict and not shared:
                        any_conflict_only = True
                if any_conflict_only and not any_group_lex:
                    unassigned_reason.setdefault(ref, "lexical_conflict")
                else:
                    unassigned_reason.setdefault(ref, "no_compatible_group")
            continue

        candidates.sort(key=lambda row: (-row[0], row[1]))
        best_score, _best_gi, best_group = candidates[0]
        if len(candidates) >= 2:
            second = candidates[1][0]
            # Dual-anchor nodes that fit 2+ groups are always ambiguous:
            # they contain both sides of a discovered conflict n-gram pair.
            # Otherwise require a clear score gap.
            if ref in dual_anchor or abs(best_score - second) <= ambiguity_margin:
                unassigned_reason[ref] = "ambiguous_multiple_groups"
                diag.ambiguous_count += 1
                continue
        if _try_join(ref, best_group, semantic_floor=rescue_floor):
            diag.rescue_assignment_count += 1
        else:
            unassigned_reason.setdefault(ref, "no_compatible_group")

    # Materialize.
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

    # Expose discovered anchors in algorithm_config (diagnostics only).
    config["group_anchor_ngrams"] = {
        g.group_ref: sorted(compute_group_anchors({m.ref for m in g.members}, texts_by_ref))
        for g in out_groups
    }
    return out_groups, unassigned, diag, config
