from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.core.embedding.factory import create_embedding_provider
from app.core.vector.postgres import PostgresVectorStore
from app.modules.topic.analyzer import TopicAnalyzer
from app.modules.topic.contracts import TopicAnalysisRequest
from app.modules.topic.repository import TopicAnalysisRepository
from app.storage.database import Database


def _rss_mb() -> float:
    """Current process RSS in MiB (Linux /proc; fallback 0 on unsupported hosts)."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                kb = float(line.split()[1])
                return kb / 1024.0
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if usage > 10_000_000:
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:  # noqa: BLE001
        return 0.0


def _pct(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROOT / "tests" / "fixtures" / "topic_vi_keywords.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "local" / "topic_benchmark_report.json"),
    )
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    if args.threshold is not None:
        overrides["TOPIC_CLUSTER_SIMILARITY_THRESHOLD"] = args.threshold
    # Prefer process env (Compose sets POSTGRES_HOST=postgres). Host CLI can export
    # POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5433 MODEL_CACHE_DIR=.models.
    settings = Settings(**overrides) if overrides else Settings()  # type: ignore[arg-type]

    request = TopicAnalysisRequest.model_validate(
        json.loads(Path(args.input).read_text(encoding="utf-8"))
    )

    database = Database(settings)
    database.wait_until_ready()
    database.migrate()
    provider = create_embedding_provider(settings)

    rss_before_load = _rss_mb()
    peak = rss_before_load
    provider.load()
    rss_loaded = _rss_mb()
    peak = max(peak, rss_loaded)

    with database.connection() as conn:
        analyzer = TopicAnalyzer(
            settings=settings,
            embedding=provider,
            vectors=PostgresVectorStore(conn),
            repository=TopicAnalysisRepository(conn),
        )
        t0 = time.perf_counter()
        result = analyzer.analyze(request)
        wall_s = time.perf_counter() - t0
        peak = max(peak, _rss_mb())
    rss_after = _rss_mb()
    peak = max(peak, rss_after)

    sizes = [g.member_count for g in result.groups]
    strong = sorted(result.groups, key=lambda g: g.mean_similarity, reverse=True)[:10]
    weak = sorted(result.groups, key=lambda g: g.min_similarity)[:10]
    questionable = []
    for group in result.groups:
        for member in group.members:
            if member.is_representative:
                continue
            if member.confidence < settings.topic_low_confidence_score:
                questionable.append(
                    {
                        "group": group.suggested_label,
                        "keyword": member.text,
                        "similarity_score": member.similarity_score,
                        "confidence": member.confidence,
                    }
                )
    questionable = sorted(questionable, key=lambda row: row["confidence"])[:20]

    report = {
        "dataset": {
            "path": str(args.input),
            "source": "synthetic_representative_vi_fixture",
            "note": "Not a private site dump. Real-site benchmark still pending if MySQL export unavailable.",
            "keyword_count": result.diagnostics.keyword_count,
        },
        "config": {
            "similarity_threshold": settings.topic_cluster_similarity_threshold,
            "min_member_similarity": settings.topic_min_member_similarity,
            "assignment_min_score": settings.topic_assignment_min_score,
            "min_group_size": settings.topic_min_group_size,
            "model": result.model.model_dump(),
            "algorithm": result.diagnostics.algorithm,
        },
        "summary": {
            "groups": result.diagnostics.group_count,
            "unassigned": result.diagnostics.unassigned_count,
            "singletons": result.diagnostics.singleton_count,
            "low_confidence_members": result.diagnostics.low_confidence_member_count,
            "group_size_p50": _pct(sizes, 0.5),
            "group_size_p90": _pct(sizes, 0.9),
            "largest_group": max(sizes) if sizes else 0,
            "histogram": result.diagnostics.group_size_histogram,
        },
        "performance": {
            "embed_ms": result.diagnostics.timings_ms.get("embed_ms"),
            "cluster_ms": result.diagnostics.timings_ms.get("cluster_ms"),
            "total_ms": result.duration_ms,
            "wall_seconds": round(wall_s, 3),
        },
        "memory_mb_rss": {
            "before_model_load": round(rss_before_load, 1),
            "after_model_load": round(rss_loaded, 1),
            "peak_during_or_after_analysis": round(peak, 1),
            "after_analysis": round(rss_after, 1),
            "note": "VmRSS from /proc/self/status inside Linux/Docker when available",
        },
        "samples": {
            "strong_groups": [
                {
                    "label": g.suggested_label,
                    "size": g.member_count,
                    "mean_similarity": g.mean_similarity,
                    "min_similarity": g.min_similarity,
                    "members": [m.text for m in g.members],
                }
                for g in strong
            ],
            "weak_groups": [
                {
                    "label": g.suggested_label,
                    "size": g.member_count,
                    "mean_similarity": g.mean_similarity,
                    "min_similarity": g.min_similarity,
                    "members": [m.text for m in g.members],
                }
                for g in weak
            ],
            "questionable_assignments": questionable,
            "unassigned_sample": [u.text for u in result.unassigned[:20]],
        },
        "analysis_id": result.analysis_id,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("TOPIC BENCHMARK")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("performance", report["performance"])
    print("memory_mb_rss", report["memory_mb_rss"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
