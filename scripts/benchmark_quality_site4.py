from __future__ import annotations

"""Compare clustering candidates on the site4 export (local / ignored).

Usage (host venv or container with dataset mounted):
  python scripts/benchmark_quality_site4.py
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.config import Settings
from app.core.clustering import (
    ClusterPoint,
    CosineAverageLinkageClusterer,
    CosineThresholdClusterer,
    CosineThresholdGreedyMedoidV2,
)
from app.core.embedding.providers.onnx_fastembed import OnnxFastEmbedProvider
from app.core.text.normalization import normalize_text
from app.modules.topic.grouping import materialize_groups_with_scores
from app.modules.topic.scoring import heuristic_confidence

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "artifacts" / "local" / "site4_keywords.json"
OUT_DIR = ROOT / "artifacts" / "local" / "quality_task31"


@dataclass
class RunSummary:
    name: str
    algorithm: str
    threshold: float
    member_threshold: float | None
    assignment_min: float
    groups: int
    unassigned: int
    singletons: int
    low_confidence: int
    largest: list[int]
    p50: float
    p90: float
    cluster_ms: int
    peak_rss_mb: float
    largest_labels: list[str]


def rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import psutil  # optional

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return -1.0


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * p))
    return float(arr[idx])


def load_points(provider: OnnxFastEmbedProvider) -> tuple[list[ClusterPoint], dict[str, str]]:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    keywords = payload["keywords"]
    texts = {str(k["ref"]): normalize_text(str(k["text"])) for k in keywords}
    ordered = sorted(texts.items(), key=lambda item: item[0])
    vectors = provider.embed_batch([text for _, text in ordered])
    points = [
        ClusterPoint(ref=ref, vector=tuple(float(x) for x in vec.vector))
        for (ref, _), vec in zip(ordered, vectors, strict=True)
    ]
    return points, texts


def scores_to_rep(points: list[ClusterPoint], result) -> dict[str, dict[str, float]]:
    by_ref = {p.ref: np.asarray(p.vector, dtype=np.float64) for p in points}
    out: dict[str, dict[str, float]] = {}
    for group in result.groups:
        rep = by_ref[group.representative_ref]
        rep = rep / np.linalg.norm(rep)
        mapping: dict[str, float] = {}
        for ref in group.member_refs:
            vec = by_ref[ref]
            vec = vec / np.linalg.norm(vec)
            mapping[ref] = float(np.dot(rep, vec))
        out[group.group_key] = mapping
    return out


def run_one(name: str, clusterer, points, texts, assignment_min: float, low_conf: float) -> RunSummary:
    settings = SimpleNamespace(
        topic_assignment_min_score=assignment_min,
        topic_low_confidence_score=low_conf,
        topic_min_group_size=2,
    )
    t0 = time.perf_counter()
    result = clusterer.cluster(points)
    cluster_ms = int((time.perf_counter() - t0) * 1000)
    score_map = scores_to_rep(points, result)
    groups, unassigned, low_c = materialize_groups_with_scores(
        cluster_result=result,
        texts_by_ref=texts,
        score_to_rep=score_map,
        settings=settings,  # type: ignore[arg-type]
    )
    sizes = sorted((g.member_count for g in groups), reverse=True)
    labels = [g.suggested_label for g in sorted(groups, key=lambda g: -g.member_count)[:10]]
    return RunSummary(
        name=name,
        algorithm=result.algorithm,
        threshold=float(getattr(clusterer, "similarity_threshold", getattr(clusterer, "seed_density_threshold", 0))),
        member_threshold=float(getattr(clusterer, "member_similarity_threshold", getattr(clusterer, "min_member_similarity", 0) or 0))
        if hasattr(clusterer, "member_similarity_threshold") or hasattr(clusterer, "min_member_similarity")
        else None,
        assignment_min=assignment_min,
        groups=len(groups),
        unassigned=len(unassigned),
        singletons=len(unassigned),
        low_confidence=low_c,
        largest=sizes[:10],
        p50=percentile(sizes, 0.5),
        p90=percentile(sizes, 0.9),
        cluster_ms=cluster_ms,
        peak_rss_mb=round(rss_mb(), 1),
        largest_labels=labels,
    )


def diagnose_baseline(points, texts) -> dict:
    """Inspect the greedy-v1 0.70 star cluster that produced ~260 members."""
    clusterer = CosineThresholdClusterer(
        similarity_threshold=0.70,
        min_member_similarity=0.62,
        min_group_size=2,
    )
    result = clusterer.cluster(points)
    biggest = max(result.groups, key=lambda g: len(g.member_refs))
    by_ref = {p.ref: np.asarray(p.vector, dtype=np.float64) for p in points}
    for ref, vec in list(by_ref.items()):
        by_ref[ref] = vec / np.linalg.norm(vec)
    rep = by_ref[biggest.representative_ref]
    sims = sorted(
        ((ref, float(np.dot(rep, by_ref[ref]))) for ref in biggest.member_refs),
        key=lambda item: -item[1],
    )
    vals = [s for _, s in sims]
    # Sample pairwise among 40 members (head/mid/tail)
    sample_refs = [sims[i][0] for i in [0, 1, 2, len(sims)//4, len(sims)//2, 3*len(sims)//4, -3, -2, -1] if abs(i) < len(sims)]
    pair = []
    for i, a in enumerate(sample_refs):
        for b in sample_refs[i + 1 :]:
            pair.append(float(np.dot(by_ref[a], by_ref[b])))
    return {
        "algorithm": result.algorithm,
        "group_size": len(biggest.member_refs),
        "representative": texts[biggest.representative_ref],
        "mean_sim_to_rep": float(np.mean(vals)),
        "min_sim_to_rep": float(np.min(vals)),
        "p10_sim": float(np.percentile(vals, 10)),
        "p50_sim": float(np.percentile(vals, 50)),
        "p90_sim": float(np.percentile(vals, 90)),
        "sample_pairwise_mean": float(np.mean(pair)) if pair else None,
        "sample_pairwise_min": float(np.min(pair)) if pair else None,
        "head": [{"text": texts[r], "sim": round(s, 4)} for r, s in sims[:8]],
        "mid": [{"text": texts[r], "sim": round(s, 4)} for r, s in sims[len(sims)//2 : len(sims)//2 + 8]],
        "tail": [{"text": texts[r], "sim": round(s, 4)} for r, s in sims[-8:]],
        "low_conf_vs_0_70": sum(1 for _, s in sims if heuristic_confidence(s, 0.70) < 0.35 and s < 1.0),
        "note": (
            "Star-shaped greedy: members only need sim>=0.70 to hub seed; "
            "min_member=0.62 is dead. Mid/tail samples mix bag/product intents."
        ),
    }


def main() -> None:
    if not DATASET.exists():
        raise SystemExit(f"missing dataset: {DATASET}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODEL_CACHE_DIR", str(ROOT / ".models"))
    settings = Settings()
    provider = OnnxFastEmbedProvider(
        model_name=settings.embedding_model,
        cache_dir=settings.model_cache_dir,
    )
    print("embedding…", flush=True)
    before = rss_mb()
    points, texts = load_points(provider)
    after_embed = rss_mb()
    print(f"points={len(points)} rss_before={before:.1f} after_embed={after_embed:.1f}", flush=True)

    diagnosis = diagnose_baseline(points, texts)
    (OUT_DIR / "diagnose_260.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "diagnosis written size=",
        diagnosis["group_size"],
        "rep=",
        diagnosis["representative"].encode("ascii", "replace").decode("ascii"),
        flush=True,
    )

    candidates: list[tuple[str, object, float]] = []
    # A: corrected greedy v2
    for member in (0.74, 0.77, 0.80):
        candidates.append(
            (
                f"greedy_v2_density0.70_member{member}",
                CosineThresholdGreedyMedoidV2(
                    seed_density_threshold=0.70,
                    member_similarity_threshold=member,
                    min_group_size=2,
                ),
                0.70,
            )
        )
    # B: average linkage
    for thr in (0.70, 0.74, 0.77, 0.80):
        candidates.append(
            (
                f"avg_linkage_{thr}",
                CosineAverageLinkageClusterer(similarity_threshold=thr, min_group_size=2),
                thr,
            )
        )
    # baseline frozen
    candidates.insert(
        0,
        (
            "baseline_greedy_v1_0.70_0.62",
            CosineThresholdClusterer(similarity_threshold=0.70, min_member_similarity=0.62),
            0.62,
        ),
    )

    summaries = []
    for name, clusterer, assign in candidates:
        print("run", name, flush=True)
        if "greedy_v2" in name:
            assign = float(clusterer.member_similarity_threshold)
        elif "baseline" in name:
            assign = 0.62
        summary = run_one(name, clusterer, points, texts, assignment_min=assign, low_conf=0.35)
        summaries.append(asdict(summary))
        print(
            f"  groups={summary.groups} unassigned={summary.unassigned} "
            f"low={summary.low_confidence} largest={summary.largest[:5]} ms={summary.cluster_ms}",
            flush=True,
        )

    (OUT_DIR / "comparison.json").write_text(
        json.dumps({"summaries": summaries, "rss_after_embed_mb": after_embed}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("wrote", OUT_DIR / "comparison.json")


if __name__ == "__main__":
    main()
