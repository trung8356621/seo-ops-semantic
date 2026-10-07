from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.config import Settings
from app.core.clustering import ClusterPoint, CosineAverageLinkageClusterer
from app.core.embedding.providers.onnx_fastembed import OnnxFastEmbedProvider
from app.core.text.normalization import normalize_text
from app.modules.topic.grouping import materialize_groups_with_scores

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "artifacts" / "local" / "site4_keywords.json"
OUT = ROOT / "artifacts" / "local" / "quality_task31" / "selected_samples.json"


def main() -> None:
    settings = Settings()
    provider = OnnxFastEmbedProvider(settings.embedding_model, settings.model_cache_dir)
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    texts = {str(k["ref"]): normalize_text(str(k["text"])) for k in payload["keywords"]}
    ordered = sorted(texts.items(), key=lambda i: i[0])
    vectors = provider.embed_batch([t for _, t in ordered])
    points = [
        ClusterPoint(ref=ref, vector=tuple(float(x) for x in vec.vector))
        for (ref, _), vec in zip(ordered, vectors, strict=True)
    ]
    clusterer = CosineAverageLinkageClusterer(similarity_threshold=0.74, min_group_size=2)
    result = clusterer.cluster(points)
    by_ref = {p.ref: np.asarray(p.vector, dtype=np.float64) for p in points}
    for ref, vec in list(by_ref.items()):
        by_ref[ref] = vec / np.linalg.norm(vec)
    score_map = {}
    for group in result.groups:
        rep = by_ref[group.representative_ref]
        score_map[group.group_key] = {
            ref: float(np.dot(rep, by_ref[ref])) for ref in group.member_refs
        }
    mat_settings = SimpleNamespace(
        topic_assignment_min_score=0.74,
        topic_low_confidence_score=0.35,
        topic_min_group_size=2,
    )
    groups, unassigned, low = materialize_groups_with_scores(
        cluster_result=result,
        texts_by_ref=texts,
        score_to_rep=score_map,
        settings=mat_settings,  # type: ignore[arg-type]
    )
    by_cohesion = sorted(groups, key=lambda g: -g.cohesion)
    by_size = sorted(groups, key=lambda g: -g.member_count)
    by_weak = sorted(groups, key=lambda g: (g.min_similarity, g.cohesion))

    def pack(g, member_limit: int = 12):
        return {
            "label": g.suggested_label,
            "size": g.member_count,
            "mean": g.mean_similarity,
            "min": g.min_similarity,
            "cohesion": g.cohesion,
            "members": [
                {"text": m.text, "sim": m.similarity_score, "conf": m.confidence}
                for m in sorted(g.members, key=lambda m: -m.similarity_score)[:member_limit]
            ],
        }

    suspicious = []
    for g in by_size[:15]:
        for m in g.members:
            if m.is_representative:
                continue
            if m.similarity_score < 0.78 or m.confidence < 0.2:
                suspicious.append(
                    {
                        "group": g.suggested_label,
                        "text": m.text,
                        "sim": m.similarity_score,
                        "conf": m.confidence,
                    }
                )
    suspicious = sorted(suspicious, key=lambda r: r["sim"])[:20]

    near_boundary = []
    for g in groups:
        for m in g.members:
            if 0.74 <= m.similarity_score <= 0.78 and not m.is_representative:
                near_boundary.append(
                    {
                        "group": g.suggested_label,
                        "text": m.text,
                        "sim": m.similarity_score,
                        "conf": m.confidence,
                    }
                )
    near_boundary = near_boundary[:20]

    OUT.write_text(
        json.dumps(
            {
                "summary": {
                    "groups": len(groups),
                    "unassigned": len(unassigned),
                    "low_confidence": low,
                    "largest": [g.member_count for g in by_size[:10]],
                },
                "strongest": [pack(g) for g in by_cohesion[:10]],
                "weakest": [pack(g) for g in by_weak[:10]],
                "largest": [pack(g, 20) for g in by_size[:10]],
                "suspicious": suspicious,
                "near_boundary": near_boundary,
                "former_hub_label_present": any(
                    "túi đeo chéo" in g.suggested_label.lower() for g in groups
                ),
                "former_hub_sizes": [
                    g.member_count
                    for g in groups
                    if "túi đeo chéo" in g.suggested_label.lower()
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
