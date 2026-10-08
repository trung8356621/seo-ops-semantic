from __future__ import annotations

from typing import Sequence

import re

from app.core.embedding.contracts import EmbeddingProvider
from app.core.similarity.cosine import cosine_similarity
from app.core.text.concept_lexical import matches_value
from app.core.text.normalization import normalize_text
from app.modules.topic_group_retrieval.contracts import (
    TopicGroupCandidateIn,
    TopicGroupEvidenceOut,
    TopicGroupMatchOut,
    TopicGroupMatchRequest,
    TopicGroupMatchResponse,
)


def _is_cta_or_boilerplate(text: str) -> bool:
    t = text.lower().strip()
    if len(t) > 160:
        return False
    return bool(re.search(r"(hotline|email|gọi ngay|đặt lịch|liên hệ|inbox|nhận mẫu.*miễn phí|\b0\d{8,10}\b)", t))


def _is_heading(text: str) -> bool:
    t = text.strip()
    return 15 <= len(t) <= 120 and not t.endswith((".", ":", "!", "?", ";", ","))


def segment_content(text: str, max_segments: int = 12) -> list[str]:
    """Extract bounded meaningful segments from article content or short queries."""
    clean = text.strip()
    if not clean:
        return []
    # Short query (< 300 chars, no multi-paragraphs): keep as single segment
    if len(clean) <= 300 and clean.count("\n") <= 1:
        return [clean]

    lines = [p.strip() for p in clean.split("\n") if p.strip()]
    content_lines = [p for p in lines if not _is_cta_or_boilerplate(p)]
    if not content_lines:
        content_lines = lines

    if len(content_lines) <= max_segments:
        return content_lines

    # Priority:
    # 1. Lead lines (first 2 content lines establish article context)
    # 2. Section headings (structural topic signals)
    # 3. Sample from remaining body lines across the text
    lead = content_lines[:2]
    rest = content_lines[2:]

    headings = [p for p in rest if _is_heading(p)]
    bodies = [p for p in rest if not _is_heading(p)]

    selected = list(lead)
    for h in headings:
        if len(selected) < max_segments:
            selected.append(h)

    remaining_space = max_segments - len(selected)
    if remaining_space > 0 and bodies:
        if len(bodies) <= remaining_space:
            selected.extend(bodies)
        else:
            step = len(bodies) / remaining_space
            for i in range(remaining_space):
                selected.append(bodies[int(i * step)])

    return selected


class TopicGroupMatcher:
    def __init__(self, embedding: EmbeddingProvider) -> None:
        self._embedding = embedding

    def match(self, request: TopicGroupMatchRequest) -> TopicGroupMatchResponse:
        segments = segment_content(request.query, max_segments=12)
        if not segments or not request.groups:
            return TopicGroupMatchResponse(scope_ref=request.scope_ref, matches=[])

        # Collect unique strings to embed in a single batch
        unique_texts: list[str] = []
        seen: set[str] = set()

        for seg in segments:
            norm_seg = normalize_text(seg)
            if norm_seg and norm_seg not in seen:
                seen.add(norm_seg)
                unique_texts.append(norm_seg)

        for group in request.groups:
            targets = _group_targets(group)
            for target in targets:
                norm_tgt = normalize_text(target)
                if norm_tgt and norm_tgt not in seen:
                    seen.add(norm_tgt)
                    unique_texts.append(norm_tgt)

        vectors: dict[str, tuple[float, ...]] = {}
        if unique_texts:
            embed_results = self._embedding.embed_batch(unique_texts)
            for text, res in zip(unique_texts, embed_results, strict=True):
                vectors[text] = res.vector

        ranked: list[TopicGroupMatchOut] = []
        query_text = request.query

        for group in request.groups:
            scored = self._score_candidate(
                group=group,
                query_text=query_text,
                segments=segments,
                vectors=vectors,
            )
            if scored.score < request.policy.min_score:
                continue
            ranked.append(scored)

        ranked.sort(key=lambda item: item.score, reverse=True)
        return TopicGroupMatchResponse(
            scope_ref=request.scope_ref,
            matches=ranked[: request.policy.limit],
        )

    def _score_candidate(
        self,
        *,
        group: TopicGroupCandidateIn,
        query_text: str,
        segments: Sequence[str],
        vectors: dict[str, tuple[float, ...]],
    ) -> TopicGroupMatchOut:
        targets = _group_targets(group)
        target_vectors = [
            vectors[normalize_text(t)] for t in targets if normalize_text(t) in vectors
        ]

        # Compute cosine similarity across segments
        seg_sims: list[float] = []
        best_overall_sim = -1.0
        best_example = targets[0] if targets else ""
        best_seg_idx = 0

        for s_idx, seg in enumerate(segments):
            seg_norm = normalize_text(seg)
            s_vec = vectors.get(seg_norm)
            if s_vec is None:
                seg_sims.append(0.0)
                continue

            seg_best = -1.0
            seg_best_target = targets[0] if targets else ""
            for target, t_vec in zip(targets, target_vectors, strict=False):
                sim = cosine_similarity(s_vec, t_vec)
                if sim > seg_best:
                    seg_best = sim
                    seg_best_target = target

            seg_sims.append(max(0.0, seg_best))
            if seg_best > best_overall_sim:
                best_overall_sim = seg_best
                best_example = seg_best_target
                best_seg_idx = s_idx

        # Aggregate semantic score across segments
        if len(segments) <= 1:
            sem_score = max(0.0, best_overall_sim)
        else:
            sorted_sims = sorted(seg_sims, reverse=True)
            top1 = sorted_sims[0]
            half_k = max(1, len(sorted_sims) // 2)
            top_half_mean = sum(sorted_sims[:half_k]) / half_k
            lead_sim = max(seg_sims[:2]) if len(seg_sims) >= 2 else seg_sims[0]
            # Blend peak relevance, sustained topic coverage, and lead relevance
            sem_score = 0.45 * top1 + 0.35 * top_half_mean + 0.20 * lead_sim

        sem_score = max(0.0, min(1.0, sem_score))

        # Lexical matching
        label = group.label.strip()
        exact_query_match = False
        label_in_text = False
        if label:
            exact_query_match = matches_value(query_text, label, "exact")
            label_in_text = matches_value(query_text, label, "phrase")

        example_in_text = False
        matched_example: str | None = None
        for ex in group.examples:
            if ex.strip() and matches_value(query_text, ex, "phrase"):
                example_in_text = True
                matched_example = ex
                break

        lexical_matched = exact_query_match or label_in_text or example_in_text

        # Scoring policy:
        # - Exact query name match receives deterministic top priority.
        # - Label / example in text provides bounded boost proportional to semantic relevance.
        # - Incidental mentions with weak semantic support receive no boost.
        if exact_query_match:
            final_score = max(0.95, sem_score)
        elif label_in_text:
            factor = max(0.0, min(1.0, (sem_score - 0.40) / 0.40))
            boost = 0.08 * factor
            final_score = min(1.0, sem_score + boost)
        elif example_in_text:
            factor = max(0.0, min(1.0, (sem_score - 0.45) / 0.40))
            boost = 0.04 * factor
            final_score = min(1.0, sem_score + boost)
        else:
            final_score = sem_score

        best_seg_text = segments[best_seg_idx] if segments else None

        return TopicGroupMatchOut(
            ref=group.ref,
            score=round(final_score, 4),
            evidence=TopicGroupEvidenceOut(
                lexical_matched=lexical_matched,
                semantic_score=round(sem_score, 4),
                best_example=best_example or matched_example,
                best_segment=best_seg_text[:120] if best_seg_text else None,
                exact_name_matched=exact_query_match,
            ),
        )


def _group_targets(group: TopicGroupCandidateIn) -> list[str]:
    label = group.label.strip()
    seen: set[str] = set()
    targets: list[str] = []
    if label:
        seen.add(normalize_text(label))
        targets.append(label)
    for ex in group.examples:
        norm = normalize_text(ex)
        if norm and norm not in seen:
            seen.add(norm)
            targets.append(ex)
    return targets
