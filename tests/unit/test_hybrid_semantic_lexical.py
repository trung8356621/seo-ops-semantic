from __future__ import annotations

from typing import Sequence

import numpy as np

from app.config import Settings
from app.core.clustering import ClusterPoint
from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text
from app.modules.keyword_grouping.analyzer import KeywordGroupAnalyzer
from app.modules.keyword_grouping.contracts import KeywordGroupAnalysisRequest, KeywordIn
from app.modules.keyword_grouping.hybrid import run_hybrid_semantic_lexical_v1


def _settings(**overrides: object) -> Settings:
    base = {
        "KEYWORD_GROUP_ALGORITHM": "hybrid_semantic_lexical_v1",
        "KEYWORD_GROUP_SEMANTIC_FLOOR": 0.55,
        "KEYWORD_GROUP_RESCUE_SEMANTIC_FLOOR": 0.50,
        "KEYWORD_GROUP_CONTAINMENT_MIN": 0.67,
        "KEYWORD_GROUP_MIN_GROUP_SIZE": 2,
        "TOPIC_EMBEDDING_CACHE_ENABLED": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _unit(vec: Sequence[float]) -> tuple[float, ...]:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr)) or 1.0
    return tuple(float(x / norm) for x in arr)


class ScriptedEmbedding:
    """High cross-family cosine on purpose — lexical must still split."""

    PROVIDER_KEY = "scripted"
    MODEL_KEY = "scripted"
    MODEL_VERSION = "v1"
    DIMENSIONS = 8

    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self._vectors = {normalize_text(k): v for k, v in vectors.items()}
        self.calls = 0

    @property
    def provider_key(self) -> str:
        return self.PROVIDER_KEY

    @property
    def model_key(self) -> str:
        return self.MODEL_KEY

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION

    @property
    def dimensions(self) -> int | None:
        return self.DIMENSIONS

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        self.calls += 1
        out: list[EmbeddingResult] = []
        for text in texts:
            key = normalize_text(text)
            vector = self._vectors[key]
            out.append(
                EmbeddingResult(
                    vector=vector,
                    model_key=self.MODEL_KEY,
                    model_version=self.MODEL_VERSION,
                    dimensions=self.DIMENSIONS,
                    provider=self.PROVIDER_KEY,
                )
            )
        return out


def _group_member_sets(groups) -> list[set[str]]:  # noqa: ANN001
    return [ {m.ref for m in g.members} for g in groups ]


def test_case_a_school_student_split_despite_high_cosine() -> None:
    # All balo* near the same semantic pole (simulates Postman ~0.93–0.97).
    balo_pole = _unit([1.0, 0.08, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    bag_pole = _unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    suitcase_pole = _unit([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    texts = {
        "1": "balo học sinh",
        "2": "balo cho học sinh",
        "3": "balo sinh viên",
        "4": "balo cho sinh viên",
        "5": "túi canvas",
        "6": "vali kéo du lịch",
    }
    vectors = {
        texts["1"]: balo_pole,
        texts["2"]: _unit([1.0, 0.09, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        texts["3"]: _unit([1.0, 0.07, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0]),
        texts["4"]: _unit([1.0, 0.075, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0]),
        texts["5"]: bag_pole,
        texts["6"]: suitcase_pole,
    }
    points = [ClusterPoint(ref=ref, vector=vectors[text]) for ref, text in texts.items()]
    groups, unassigned, diag, _cfg = run_hybrid_semantic_lexical_v1(
        points=points,
        texts_by_ref=texts,
        settings=_settings(),
    )
    member_sets = _group_member_sets(groups)
    assert {"1", "2"} in member_sets
    assert {"3", "4"} in member_sets
    assert len(groups) == 2
    un_refs = {u.ref for u in unassigned}
    assert un_refs == {"5", "6"}
    # Critical: families must not merge.
    for members in member_sets:
        assert not ({"1", "3"} <= members)
        assert not ({"2", "4"} <= members)
    assert diag.lexical_reject_count >= 1
    assert diag.strategy == "hybrid_semantic_lexical_v1"


def test_case_b_recall_and_no_bridge() -> None:
    school = _unit([1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    student = _unit([0.05, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Bridge sits between poles (high sim to both families).
    bridge = _unit([0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    control_a = _unit([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    control_b = _unit([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    control_c = _unit([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    keywords = [
        ("hs-01", "balo học sinh", _unit([1.0, 0.04, 0.01, 0, 0, 0, 0, 0])),
        ("hs-02", "cặp học sinh", _unit([0.98, 0.06, 0.02, 0, 0, 0, 0, 0])),
        ("hs-03", "giúp bạn chọn mua balo học sinh", _unit([0.97, 0.05, 0.03, 0, 0, 0, 0, 0])),
        ("hs-04", "xưởng may balo học sinh giá tốt nhất tại Hợp Phát", _unit([0.96, 0.07, 0.02, 0, 0, 0, 0, 0])),
        ("hs-05", "xưởng may cặp học sinh cấp 1 tại Hợp Phát", _unit([0.95, 0.08, 0.01, 0, 0, 0, 0, 0])),
        ("hs-06", "xưởng may balo học sinh tại Hợp Phát", _unit([0.99, 0.03, 0.02, 0, 0, 0, 0, 0])),
        ("mix-01", "xưởng sản xuất balo học sinh - sinh viên tại tphcm", bridge),
        ("sv-01", "balo sinh viên", _unit([0.04, 1.0, 0.01, 0, 0, 0, 0, 0])),
        ("sv-02", "balo giá rẻ cho sinh viên", _unit([0.06, 0.98, 0.02, 0, 0, 0, 0, 0])),
        ("sv-03", "balo cho sinh viên đại học", _unit([0.05, 0.97, 0.03, 0, 0, 0, 0, 0])),
        ("ctrl-01", "túi canvas", control_a),
        ("ctrl-02", "balo laptop", control_b),
        ("ctrl-03", "vali kéo du lịch", control_c),
    ]
    # Ensure school/student poles are used (lint silence for unused).
    assert school and student

    texts = {ref: text for ref, text, _vec in keywords}
    points = [ClusterPoint(ref=ref, vector=vec) for ref, _text, vec in keywords]
    groups, unassigned, _diag, _cfg = run_hybrid_semantic_lexical_v1(
        points=points,
        texts_by_ref=texts,
        settings=_settings(),
    )

    by_ref = {m.ref: g for g in groups for m in g.members}
    hs_refs = {f"hs-0{i}" for i in range(1, 7)}
    sv_refs = {"sv-01", "sv-02", "sv-03"}

    # School family materially grouped together.
    hs_groups = {id(by_ref[r]) for r in hs_refs if r in by_ref}
    assert len(hs_groups) == 1
    school_group = next(g for g in groups if any(m.ref.startswith("hs-") for m in g.members))
    school_members = {m.ref for m in school_group.members}
    assert len(school_members & hs_refs) >= 5

    # Student family complete and separate.
    student_group = next(g for g in groups if any(m.ref.startswith("sv-") for m in g.members))
    student_members = {m.ref for m in student_group.members}
    assert sv_refs <= student_members
    assert school_members.isdisjoint(student_members)

    un_map = {u.ref: u.reason for u in unassigned}
    assert "mix-01" in un_map
    assert un_map["mix-01"] == "ambiguous_multiple_groups"
    for ctrl in ("ctrl-01", "ctrl-02", "ctrl-03"):
        assert ctrl in un_map
        assert ctrl not in school_members
        assert ctrl not in student_members


def test_property_distinct_modifiers_not_merged_by_cosine_alone() -> None:
    """Synthetic product+modifierA vs product+modifierB — no phrase hardcoding."""
    pole = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    texts = {
        "a1": "widget alpha beta",
        "a2": "widget cho alpha beta",
        "b1": "widget gamma delta",
        "b2": "widget cho gamma delta",
        "z": "totally unrelated phrase",
    }
    points = [
        ClusterPoint(ref="a1", vector=pole),
        ClusterPoint(ref="a2", vector=_unit([0.99, 0.01, 0, 0, 0, 0, 0, 0])),
        ClusterPoint(ref="b1", vector=_unit([0.98, 0.02, 0, 0, 0, 0, 0, 0])),
        ClusterPoint(ref="b2", vector=_unit([0.97, 0.03, 0, 0, 0, 0, 0, 0])),
        ClusterPoint(ref="z", vector=_unit([0, 0, 1, 0, 0, 0, 0, 0])),
    ]
    groups, unassigned, _diag, _cfg = run_hybrid_semantic_lexical_v1(
        points=points,
        texts_by_ref=texts,
        settings=_settings(),
    )
    member_sets = _group_member_sets(groups)
    assert {"a1", "a2"} in member_sets
    assert {"b1", "b2"} in member_sets
    for members in member_sets:
        assert not ({"a1", "b1"} <= members)
    assert any(u.ref == "z" for u in unassigned)


def test_property_same_modifier_with_extra_words_groups() -> None:
    pole = _unit([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    texts = {
        "1": "gadget red blue",
        "2": "best price gadget red blue store nearby",
    }
    points = [
        ClusterPoint(ref="1", vector=pole),
        ClusterPoint(ref="2", vector=_unit([0.96, 0.12, 0.02, 0, 0, 0, 0, 0])),
    ]
    groups, unassigned, _diag, _cfg = run_hybrid_semantic_lexical_v1(
        points=points,
        texts_by_ref=texts,
        settings=_settings(),
    )
    assert len(groups) == 1
    assert {m.ref for m in groups[0].members} == {"1", "2"}
    assert unassigned == []


def test_analyzer_case_a_end_to_end_and_cache_namespace() -> None:
    texts = {
        "1": "balo học sinh",
        "2": "balo cho học sinh",
        "3": "balo sinh viên",
        "4": "balo cho sinh viên",
        "5": "túi canvas",
        "6": "vali kéo du lịch",
    }
    balo = _unit([1.0, 0.08, 0.02, 0, 0, 0, 0, 0])
    embedding = ScriptedEmbedding(
        {
            texts["1"]: balo,
            texts["2"]: _unit([1.0, 0.09, 0.01, 0, 0, 0, 0, 0]),
            texts["3"]: _unit([1.0, 0.07, 0.03, 0, 0, 0, 0, 0]),
            texts["4"]: _unit([1.0, 0.075, 0.025, 0, 0, 0, 0, 0]),
            texts["5"]: _unit([0, 1, 0, 0, 0, 0, 0, 0]),
            texts["6"]: _unit([0, 0, 1, 0, 0, 0, 0, 0]),
        }
    )
    from app.core.vector.contracts import VectorRecord

    class MemoryVectorStore:
        def __init__(self) -> None:
            self.rows: dict[tuple[str, str, str], VectorRecord] = {}

        def upsert(self, record: VectorRecord) -> None:
            self.rows[(record.namespace, record.entity_ref, record.model_key)] = record

        def get(self, namespace: str, entity_ref: str, model_key: str) -> VectorRecord | None:
            return self.rows.get((namespace, entity_ref, model_key))

        def delete(self, namespace: str, entity_ref: str, model_key: str | None = None) -> int:
            return 0

        def nearest(self, **kwargs):  # noqa: ANN003
            return []

    store = MemoryVectorStore()
    analyzer = KeywordGroupAnalyzer(
        settings=_settings(TOPIC_EMBEDDING_CACHE_ENABLED=True),
        embedding=embedding,
        vectors=store,
    )
    req = KeywordGroupAnalysisRequest(
        scope_ref="site-postman-a",
        language="vi",
        keywords=[KeywordIn(ref=ref, text=text) for ref, text in texts.items()],
    )
    first = analyzer.analyze(req)
    assert first.diagnostics.algorithm == "hybrid_semantic_lexical_v1"
    assert embedding.calls == 1
    second = analyzer.analyze(req)
    assert embedding.calls == 1
    assert second.diagnostics.embedding_cache["hits"] == 6
    assert second.diagnostics.embedding_cache["misses"] == 0

    sets = _group_member_sets(first.groups)
    assert {"1", "2"} in sets
    assert {"3", "4"} in sets
