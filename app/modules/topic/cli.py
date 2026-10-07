from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.core.embedding.factory import create_embedding_provider
from app.core.vector.postgres import PostgresVectorStore
from app.modules.topic.analyzer import TopicAnalyzer
from app.modules.topic.contracts import TopicAnalysisRequest, TopicKeywordIn
from app.modules.topic.repository import TopicAnalysisRepository
from app.storage.database import Database


def _load_request(path: Path) -> TopicAnalysisRequest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        keywords = [TopicKeywordIn.model_validate(row) for row in raw]
        return TopicAnalysisRequest(site_ref="local", language="vi", keywords=keywords)
    return TopicAnalysisRequest.model_validate(raw)


def _percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def cmd_analyze(args: argparse.Namespace) -> int:
    path = Path(args.input)
    request = _load_request(path)
    settings = get_settings()
    database = Database(settings)
    database.wait_until_ready()
    database.migrate()
    provider = create_embedding_provider(settings)

    with database.connection() as conn:
        analyzer = TopicAnalyzer(
            settings=settings,
            embedding=provider,
            vectors=PostgresVectorStore(conn),
            repository=TopicAnalysisRepository(conn) if args.persist else None,
        )
        result = analyzer.analyze(request)

    sizes = [g.member_count for g in result.groups]
    print("SEO-OPS Topic Analyze")
    print(f"analysis_id        {result.analysis_id}")
    print(f"keywords           {result.diagnostics.keyword_count}")
    print(f"embed_ms           {result.diagnostics.timings_ms.get('embed_ms')}")
    print(f"cluster_ms         {result.diagnostics.timings_ms.get('cluster_ms')}")
    print(f"total_ms           {result.duration_ms}")
    print(f"groups             {result.diagnostics.group_count}")
    print(f"unassigned         {result.diagnostics.unassigned_count}")
    print(f"singletons         {result.diagnostics.singleton_count}")
    print(f"low_confidence     {result.diagnostics.low_confidence_member_count}")
    print(f"cache_hits         {result.diagnostics.embedding_cache.get('hits')}")
    print(f"cache_misses       {result.diagnostics.embedding_cache.get('misses')}")
    print(f"group_size_p50     {_percentile(sizes, 0.50):.1f}")
    print(f"group_size_p90     {_percentile(sizes, 0.90):.1f}")
    print(f"largest_group      {max(sizes) if sizes else 0}")
    print(f"histogram          {result.diagnostics.group_size_histogram}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"wrote              {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.modules.topic.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a keyword JSON dataset")
    analyze.add_argument("input", help="Path to keywords JSON")
    analyze.add_argument("--output", help="Optional JSON report path")
    analyze.add_argument("--persist", action="store_true", help="Save analysis rows")
    analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
