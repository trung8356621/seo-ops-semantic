from __future__ import annotations

from app.core.embedding.contracts import EmbeddingProvider
from app.core.similarity.cosine import cosine_similarity
from app.core.text.normalization import normalize_text
from app.modules.cta_planner.contracts import (
    CtaPlanRequest,
    CtaPlanResponse,
    CtaPlacementOut,
    LegacyClassificationOut,
    SkippedSectionOut,
)

# Prototype phrases are embedding anchors, not a placement rule dictionary.
INTENT_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "product_discovery": (
        "explore suitable products related to this topic",
        "xem các lựa chọn sản phẩm phù hợp với nội dung",
        "browse product options that match this section",
    ),
    "service_discovery": (
        "learn about a relevant service offering",
        "tìm hiểu dịch vụ phù hợp với nhu cầu trong mục này",
        "discover a service that helps with this problem",
    ),
    "comparison": (
        "compare available options before deciding",
        "so sánh các phương án trước khi chọn",
        "evaluate alternatives side by side",
    ),
    "consultation": (
        "ask for guidance choosing the right option",
        "nhờ tư vấn để chọn phương án phù hợp",
        "request advice about fit size or materials",
    ),
    "conversion": (
        "take a clear next step toward a purchase",
        "bước tiếp theo để đặt hoặc mua khi đã chọn",
        "proceed when the reader is ready to act",
    ),
}

PROMO_PROTOTYPES = (
    "contact us via zalo facebook email or visit the showroom",
    "liên hệ ngay qua zalo facebook email để được tư vấn mua",
    "call now visit our page send an email browse our website",
)

INFO_PROTOTYPES = (
    "explain product characteristics materials and how to use them",
    "giải thích đặc điểm chất liệu cách chọn và bảo quản",
    "educational comparison of facts without asking the reader to contact anyone",
)

MIN_CONFIDENCE = 0.42
MIN_GAP_WORDS = 180
HARD_CAP = 5


class CtaPlanner:
    def __init__(self, embedding: EmbeddingProvider) -> None:
        self._embedding = embedding

    def plan(self, request: CtaPlanRequest) -> CtaPlanResponse:
        if not self._embedding.is_loaded:
            self._embedding.load()

        texts = self._collect_texts(request)
        embedded = self._embedding.embed_batch(texts)
        vectors = {
            normalize_text(text): result.vector
            for text, result in zip(texts, embedded, strict=True)
        }

        scored: list[tuple[str, str, float, int]] = []
        skipped: list[SkippedSectionOut] = []
        for section in request.sections:
            content = section.content.strip()
            if content == "" or section.word_count < 25:
                skipped.append(
                    SkippedSectionOut(
                        section_id=section.section_id,
                        reason="section_too_thin",
                        best_intent=None,
                        score=0.0,
                    )
                )
                continue
            intent, score = self._best_intent(vectors, content)
            if section.word_count < 40:
                score = max(0.0, score - 0.08)
            if intent is None or score < MIN_CONFIDENCE:
                skipped.append(
                    SkippedSectionOut(
                        section_id=section.section_id,
                        reason="no_cta",
                        best_intent=intent,
                        score=round(score, 4),
                    )
                )
                continue
            scored.append((section.section_id, intent, score, section.start_word))

        placements = self._distribute(request.article_word_count, scored, skipped)
        legacy = [self._classify_legacy(vectors, row) for row in request.legacy_candidates]
        return CtaPlanResponse(
            placements=placements,
            skipped=skipped,
            legacy=legacy,
            debug={
                "min_confidence": MIN_CONFIDENCE,
                "min_gap_words": MIN_GAP_WORDS,
                "candidate_sections": len(scored),
                "placement_count": len(placements),
            },
        )

    def _collect_texts(self, request: CtaPlanRequest) -> list[str]:
        texts: list[str] = []
        for examples in INTENT_PROTOTYPES.values():
            texts.extend(examples)
        texts.extend(PROMO_PROTOTYPES)
        texts.extend(INFO_PROTOTYPES)
        for section in request.sections:
            if section.content.strip() != "":
                texts.append(section.content)
        for row in request.legacy_candidates:
            if row.text.strip() != "":
                texts.append(row.text)
        return texts

    def _best_intent(self, vectors: dict[str, tuple[float, ...]], content: str) -> tuple[str | None, float]:
        source = vectors.get(normalize_text(content))
        if source is None:
            return None, 0.0
        best_intent: str | None = None
        best = -1.0
        for intent, examples in INTENT_PROTOTYPES.items():
            score = max(self._cosine(source, vectors, example) for example in examples)
            if score > best:
                best = score
                best_intent = intent
        return best_intent, max(0.0, best)

    def _distribute(
        self,
        article_words: int,
        scored: list[tuple[str, str, float, int]],
        skipped: list[SkippedSectionOut],
    ) -> list[CtaPlacementOut]:
        ceiling = min(HARD_CAP, max(1, article_words // 350)) if article_words > 0 else HARD_CAP
        ordered = sorted(scored, key=lambda row: row[2], reverse=True)
        chosen: list[tuple[str, str, float, int]] = []
        recent_intents: list[str] = []
        for section_id, intent, score, start_word in ordered:
            if len(chosen) >= ceiling:
                skipped.append(
                    SkippedSectionOut(
                        section_id=section_id,
                        reason="density_ceiling",
                        best_intent=intent,
                        score=round(score, 4),
                    )
                )
                continue
            if any(abs(start_word - picked[3]) < MIN_GAP_WORDS for picked in chosen):
                skipped.append(
                    SkippedSectionOut(
                        section_id=section_id,
                        reason="spacing",
                        best_intent=intent,
                        score=round(score, 4),
                    )
                )
                continue
            if recent_intents and intent == recent_intents[-1] and intent == "conversion":
                skipped.append(
                    SkippedSectionOut(
                        section_id=section_id,
                        reason="repeated_conversion",
                        best_intent=intent,
                        score=round(score, 4),
                    )
                )
                continue
            chosen.append((section_id, intent, score, start_word))
            recent_intents.append(intent)

        chosen.sort(key=lambda row: row[3])
        placements: list[CtaPlacementOut] = []
        for index, (section_id, intent, score, _start) in enumerate(chosen, start=1):
            placements.append(
                CtaPlacementOut(
                    placement_id=f"cta_{index:03d}",
                    section_id=section_id,
                    position="section_end",
                    intent=intent,  # type: ignore[arg-type]
                    confidence=round(min(1.0, max(0.0, score)), 4),
                    reason="semantic_score_with_spacing",
                )
            )
        return placements

    def _classify_legacy(self, vectors: dict[str, tuple[float, ...]], row) -> LegacyClassificationOut:
        if row.structural_signal == "mixed_paragraph":
            return LegacyClassificationOut(
                candidate_id=row.candidate_id,
                section_id=row.section_id,
                classification="uncertain",
                confidence=round(self._pole(vectors, row.text, PROMO_PROTOTYPES), 4),
                reason="mixed_content_requires_review",
            )
        promo = self._pole(vectors, row.text, PROMO_PROTOTYPES)
        info = self._pole(vectors, row.text, INFO_PROTOTYPES)
        if row.structural_signal in {"standalone_blockquote", "standalone_paragraph"} and promo >= 0.45 and promo >= info:
            classification = "promotional"
            reason = "standalone_promotional"
        elif promo < 0.35 or info > promo:
            classification = "editorial"
            reason = "editorial_or_descriptive"
        else:
            classification = "uncertain"
            reason = "ambiguous_promotional_signal"
        return LegacyClassificationOut(
            candidate_id=row.candidate_id,
            section_id=row.section_id,
            classification=classification,  # type: ignore[arg-type]
            confidence=round(max(promo, info), 4),
            reason=reason,
        )

    def _pole(self, vectors: dict[str, tuple[float, ...]], text: str, prototypes: tuple[str, ...]) -> float:
        source = vectors.get(normalize_text(text))
        if source is None or text.strip() == "":
            return 0.0
        return max(self._cosine(source, vectors, example) for example in prototypes)

    def _cosine(self, source: tuple[float, ...], vectors: dict[str, tuple[float, ...]], example: str) -> float:
        target = vectors.get(normalize_text(example))
        if target is None:
            return 0.0
        return cosine_similarity(source, target)
