"""Explicit concept decision policies — no opaque weighted lexical+semantic blend."""

from __future__ import annotations

from app.modules.concept_matching.contracts import (
    ConceptDecisionPolicy,
    LexicalEvidenceOut,
    MatchingStrategy,
)
from app.modules.concept_matching.scoring import ConceptScoreEvidence


def apply_semantic_gate(
    evidence: ConceptScoreEvidence,
    policy: ConceptDecisionPolicy,
) -> bool:
    if policy.min_positive_score is None:
        return False
    if evidence.positive_max is None:
        return False
    if evidence.positive_max < policy.min_positive_score:
        return False
    if evidence.negative_max is not None and evidence.margin is not None:
        if evidence.margin < policy.min_margin:
            return False
    return True


def decide_suggested_match(
    *,
    strategy: MatchingStrategy,
    lexical: LexicalEvidenceOut,
    semantic: ConceptScoreEvidence | None,
    policy: ConceptDecisionPolicy | None,
) -> bool | None:
    """Return suggested_match, or None when no decision_policy was supplied."""
    if policy is None:
        return None

    lexical_positive = lexical.matched and not lexical.negative_matched

    if strategy == "lexical":
        return lexical_positive

    if strategy == "semantic":
        if semantic is None:
            return False
        return apply_semantic_gate(semantic, policy)

    # hybrid: deterministic evidence wins; semantic is optional fallback.
    if lexical_positive:
        return True
    if lexical.negative_matched and lexical.matched:
        # Positive and negative both hit — do not auto-accept; optional semantic fallback.
        pass
    if not policy.semantic_fallback:
        return False
    if semantic is None:
        return False
    return apply_semantic_gate(semantic, policy)
