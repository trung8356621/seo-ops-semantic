from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from app.config import Settings
from app.core.clustering import ClusterPoint
from app.core.text.lexical import _is_contiguous_span, content_tokens, informative_ngrams, pair_evidence
from app.modules.keyword_grouping.contracts import GroupMemberOut, GroupOut, UnassignedOut
from app.modules.keyword_grouping.scoring import cohesion_score

ALGORITHM = "hybrid_semantic_lexical_v1"
RECIPROCAL_NEIGHBOR_K = 3


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
    industry_evidence_keyword_count: int = 0
    industry_membership_count: int = 0
    industry_supported_edge_count: int = 0
    industry_supported_assignment_count: int = 0


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
    shared_industry: bool = False,
    frequent_ngrams: frozenset[str] | None = None,
) -> tuple[PairRelation, object]:
    """Three-state pair relation. Cosine alone never yields COMPATIBLE.

    Shared Industry Group membership is positive support only.
    It does not override lexical conflict or the semantic floor.
    """
    evidence = pair_evidence(
        text_a,
        text_b,
        containment_min=containment_min,
        frequent_ngrams=frequent_ngrams,
    )
    if evidence.conflict:
        return PairRelation.INCOMPATIBLE, evidence
    if cosine < semantic_floor:
        return PairRelation.UNKNOWN, evidence
    if evidence.compatible:
        return PairRelation.COMPATIBLE, evidence
    if shared_industry:
        return PairRelation.COMPATIBLE, evidence
    return PairRelation.UNKNOWN, evidence


def _frequent_ngrams(texts_by_ref: Mapping[str, str]) -> frozenset[str]:
    """N-grams in at least 10% of a large inventory are not compatibility evidence."""
    count = len(texts_by_ref)
    if count < 40:
        return frozenset()
    cutoff = max(8, (count + 9) // 10)
    seen: dict[str, int] = {}
    for text in texts_by_ref.values():
        for ngram in _member_ngrams(text):
            seen[ngram] = seen.get(ngram, 0) + 1
    return frozenset(ngram for ngram, hits in seen.items() if hits >= cutoff)


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
    frequent_ngrams: frozenset[str] | None = None,
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
            frequent_ngrams=frequent_ngrams,
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
    frequent_ngrams: frozenset[str] | None = None,
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
    shared_anchors = frozenset(
        ngram for ngram in (cand_ngrams & anchors) if ngram not in (frequent_ngrams or ())
    )

    # Lexical vs representative.
    ev_rep = pair_evidence(
        texts_by_ref[ref],
        texts_by_ref[rep],
        containment_min=containment_min,
        frequent_ngrams=frequent_ngrams,
    )

    # A shared anchor must not skip a conflict with the representative.
    if ev_rep.conflict:
        conflict_hits = 0
        checked = 0
        for member in group.members:
            checked += 1
            ev = pair_evidence(
                texts_by_ref[ref],
                texts_by_ref[member],
                containment_min=containment_min,
                frequent_ngrams=frequent_ngrams,
            )
            if ev.conflict:
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
            frequent_ngrams=frequent_ngrams,
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
    memberships_by_ref: Mapping[str, frozenset[str]] | None = None,
) -> tuple[list[GroupOut], list[UnassignedOut], HybridDiagnostics, dict[str, object]]:
    """V3: reciprocal semantic neighbors, pair-local lexical veto, locked representative."""
    memberships = memberships_by_ref or {}
    diag = HybridDiagnostics(
        industry_evidence_keyword_count=sum(1 for keys in memberships.values() if keys),
        industry_membership_count=sum(len(keys) for keys in memberships.values()),
    )
    config = {
        "semantic_candidate_floor": settings.keyword_group_semantic_floor,
        "rescue_semantic_floor": settings.keyword_group_rescue_semantic_floor,
        "containment_min": settings.keyword_group_containment_min,
        "min_group_size": settings.keyword_group_min_group_size,
        "ambiguity_margin": settings.keyword_group_ambiguity_margin,
        "version": "v3",
        "reciprocal_neighbor_k": RECIPROCAL_NEIGHBOR_K,
        "unknown_seed_rule": "not_a_seed",
        "industry_evidence": bool(memberships),
        "industry_evidence_keyword_count": diag.industry_evidence_keyword_count,
        "industry_membership_count": diag.industry_membership_count,
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

    pair_started = time.perf_counter()
    relations: dict[tuple[str, str], PairRelation] = {}
    containment_by_pair: dict[tuple[str, str], float] = {}
    industry_supported_edges: set[tuple[str, str]] = set()

    for i, a in enumerate(refs):
        for b in refs[i + 1 :]:
            cosine = float(sim[i, index[b]])
            shared_keys = memberships.get(a, frozenset()) & memberships.get(b, frozenset())
            rel, evidence = classify_pair_relation(
                text_a=texts_by_ref[a],
                text_b=texts_by_ref[b],
                cosine=cosine,
                semantic_floor=floor,
                containment_min=containment_min,
                shared_industry=bool(shared_keys),
            )
            key = (a, b)
            relations[key] = rel
            containment_by_pair[key] = float(getattr(evidence, "containment", 0.0))
            if rel is PairRelation.INCOMPATIBLE and cosine >= floor:
                diag.lexical_reject_count += 1
                continue
            if rel is not PairRelation.COMPATIBLE:
                continue
            lexical_compatible = bool(getattr(evidence, "compatible", False)) and not bool(
                getattr(evidence, "conflict", False)
            )
            if shared_keys and not lexical_compatible:
                diag.industry_supported_edge_count += 1
                industry_supported_edges.add(key)
            diag.semantic_candidate_edges += 1
    pair_seconds = time.perf_counter() - pair_started

    def _neighbors(min_cosine: float) -> dict[str, list[str]]:
        ranked: dict[str, list[str]] = {}
        for ref in refs:
            row: list[tuple[float, str]] = []
            for other in refs:
                if other == ref:
                    continue
                cosine = _sim(sim, index, ref, other)
                if cosine < min_cosine:
                    continue
                if relations[_pair_key(ref, other)] is PairRelation.INCOMPATIBLE:
                    continue
                row.append((cosine, other))
            row.sort(key=lambda item: (-item[0], item[1]))
            ranked[ref] = [other for _cosine, other in row[:RECIPROCAL_NEIGHBOR_K]]
        return ranked

    def _reciprocal_seeds(pool: set[str], ranked: Mapping[str, list[str]]) -> list[tuple[float, str, str]]:
        found: list[tuple[float, str, str]] = []
        for left in sorted(pool):
            for right in ranked.get(left, []):
                if right not in pool or right <= left:
                    continue
                if left not in ranked.get(right, []):
                    continue
                key = _pair_key(left, right)
                relation = relations[key]
                if relation is not PairRelation.COMPATIBLE:
                    continue
                found.append((_sim(sim, index, left, right), left, right))
        found.sort(key=lambda item: (-item[0], item[1], item[2]))
        return found

    def _span_link(ref: str, group: _MutableGroup) -> bool:
        candidate = content_tokens(texts_by_ref[ref])
        for member in group.members:
            member_tokens = content_tokens(texts_by_ref[member])
            if len(member_tokens) >= 3 and _is_contiguous_span(candidate, member_tokens):
                return True
            if len(candidate) >= 3 and _is_contiguous_span(member_tokens, candidate):
                return True
        return False

    def _coherent(ref: str, group: _MutableGroup, ranked: Mapping[str, list[str]], min_cosine: float) -> tuple[bool, bool]:
        rep = group.representative_ref
        if rep is None or ref in group.members:
            return False, False
        if _sim(sim, index, ref, rep) < min_cosine:
            return False, False
        key = _pair_key(ref, rep)
        relation = relations[key]
        evidence = pair_evidence(
            texts_by_ref[ref],
            texts_by_ref[rep],
            containment_min=containment_min,
        )
        if relation is PairRelation.INCOMPATIBLE or evidence.conflict:
            return False, True
        if evidence.compatible or _span_link(ref, group):
            return True, False
        reciprocal = ref in ranked.get(rep, []) and rep in ranked.get(ref, [])
        if relation is PairRelation.COMPATIBLE and reciprocal and containment_by_pair[key] >= containment_min:
            return True, False
        return False, False

    seed_started = time.perf_counter()
    assigned: set[str] = set()
    ambiguous: set[str] = set()
    unassigned_reason: dict[str, str] = {}
    groups: list[_MutableGroup] = []
    primary_seed_groups = 0
    rescue_created_groups = 0

    def _open_seeds(pairs: list[tuple[float, str, str]], *, rescue: bool) -> None:
        nonlocal primary_seed_groups, rescue_created_groups
        opened = 0
        for _cosine, left, right in pairs:
            if left in assigned or right in assigned or left in ambiguous or right in ambiguous:
                continue
            group = _MutableGroup(members={left, right})
            group.representative_ref = _pick_medoid({left, right}, sim, index)
            group.anchor_ngrams = compute_group_anchors(group.members, texts_by_ref)
            assigned.add(left)
            assigned.add(right)
            groups.append(group)
            opened += 1
            if _pair_key(left, right) in industry_supported_edges:
                diag.industry_supported_assignment_count += 1
        if rescue:
            rescue_created_groups += opened
        else:
            primary_seed_groups += opened

    primary_ranked = _neighbors(floor)
    _open_seeds(_reciprocal_seeds(set(refs), primary_ranked), rescue=False)
    seed_seconds = time.perf_counter() - seed_started

    def _assign(ranked: Mapping[str, list[str]], min_cosine: float, *, rescue: bool) -> None:
        pending = [ref for ref in refs if ref not in assigned and ref not in ambiguous]
        for ref in pending:
            hits = [group for group in groups if _coherent(ref, group, ranked, min_cosine)[0]]
            if len(hits) > 1:
                ambiguous.add(ref)
                unassigned_reason[ref] = "ambiguous_multiple_groups"
                diag.ambiguous_count += 1
                continue
            if len(hits) != 1:
                continue
            group = hits[0]
            group.members.add(ref)
            group.anchor_ngrams = compute_group_anchors(group.members, texts_by_ref)
            assigned.add(ref)
            if rescue:
                diag.rescue_assignment_count += 1
            rep = group.representative_ref or ref
            if _pair_key(ref, rep) in industry_supported_edges:
                diag.industry_supported_assignment_count += 1

    growth_started = time.perf_counter()
    _assign(primary_ranked, floor, rescue=False)
    growth_seconds = time.perf_counter() - growth_started
    diag.primary_group_count = len(groups)

    rescue_started = time.perf_counter()
    rescue_ranked = _neighbors(rescue_floor)
    remaining = {ref for ref in refs if ref not in assigned and ref not in ambiguous}
    _open_seeds(_reciprocal_seeds(remaining, rescue_ranked), rescue=True)
    _assign(rescue_ranked, rescue_floor, rescue=True)
    rescue_seconds = time.perf_counter() - rescue_started

    kept: list[_MutableGroup] = []
    for group in groups:
        if len(group.members) < min_size:
            for member in group.members:
                assigned.discard(member)
                unassigned_reason.setdefault(member, "below_min_group_size")
            continue
        kept.append(group)
    groups = kept

    for ref in refs:
        if ref in assigned or ref in ambiguous:
            continue
        best = max((_sim(sim, index, ref, other) for other in refs if other != ref), default=0.0)
        if best < rescue_floor:
            unassigned_reason[ref] = "below_semantic_floor"
            continue
        saw_veto = False
        saw_open = False
        for other in refs:
            if other == ref or _sim(sim, index, ref, other) < rescue_floor:
                continue
            if relations[_pair_key(ref, other)] is PairRelation.INCOMPATIBLE:
                saw_veto = True
            else:
                saw_open = True
        if saw_veto and not saw_open:
            unassigned_reason[ref] = "lexical_conflict"
        else:
            unassigned_reason[ref] = "no_compatible_group"

    config["reciprocal_seed_count"] = primary_seed_groups + rescue_created_groups
    config["lexical_veto_count"] = diag.lexical_reject_count
    config["rescue_created_group_count"] = rescue_created_groups
    config["ambiguous_count"] = diag.ambiguous_count
    config["timing_seconds"] = {
        "pair_analysis": round(pair_seconds, 4),
        "seed_selection": round(seed_seconds, 4),
        "group_growth": round(growth_seconds, 4),
        "rescue": round(rescue_seconds, 4),
    }
    _ = ambiguity_margin

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
