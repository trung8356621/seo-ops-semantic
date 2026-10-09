"""Dry-run local semantic routing. No business writes and no paid answer model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.embedding.factory import create_embedding_provider
from app.core.similarity.cosine import cosine_similarity
from app.config import get_settings
from app.modules.tool_intent.weighted import WeightedGroup, WeightedTarget, group_relevance, rank_weighted


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Dry-run SEO Ops agent semantic routing.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--config", required=True, help="Published semantic-routing JSON.")
    parser.add_argument("--scope", choices=["auto", "global", "module"], default="auto")
    parser.add_argument("--module", default="")
    args = parser.parse_args(argv)

    document = json.loads(Path(args.config).read_text(encoding="utf-8"))
    settings = get_settings()
    provider = create_embedding_provider(settings)
    provider.load()

    if args.scope == "module":
        if args.module == "":
            print("module scope requires --module", file=sys.stderr)
            return 2
        _print_level("internal", args.module, _rank(provider, args.question, document["modules"].get(args.module, [])))
        return 0

    global_result = _rank(provider, args.question, document.get("global", []))
    _print_level("global", "modules", global_result)
    if args.scope == "global" or global_result[0] != "confident" or global_result[1] is None:
        _print_decision(global_result[1], None, None)
        return 0

    module = global_result[1]
    internal = _rank(provider, args.question, document.get("modules", {}).get(module, []))
    _print_level("internal", module, internal)
    _print_decision(module, internal[1], internal[0])
    return 0


def _rank(provider, question: str, raw_groups: list[dict]):
    groups = []
    for row in raw_groups:
        if row.get("enabled", True) is False:
            continue
        groups.append(
            WeightedGroup(
                id=str(row["id"]),
                examples=[str(item) for item in row.get("examples", []) if str(item).strip()],
                targets=[WeightedTarget(ref=str(item["ref"]), weight=float(item["weight"])) for item in row.get("targets", [])],
            )
        )
    texts = [question]
    spans = []
    for group in groups:
        start = len(texts)
        texts.extend(group.examples)
        spans.append((group.id, start, len(texts)))
    vectors = [item.vector for item in provider.embed_batch(texts)]
    query = vectors[0]
    relevances = {}
    examples = {}
    for group, (group_id, start, end) in zip(groups, spans, strict=True):
        sims = [cosine_similarity(query, vectors[index]) for index in range(start, end)]
        relevances[group_id] = group_relevance(sims)
        best = max(range(len(sims)), key=lambda index: sims[index]) if sims else 0
        examples[group_id] = group.examples[best] if sims else ""
    return rank_weighted(relevances, examples, groups)


def _print_level(scope: str, name: str, result) -> None:
    status, winner, candidates = result
    print(f"[{scope}:{name}] status={status} winner={winner or '-'}")
    for item in candidates:
        print(
            f"  {item.ref} relevance={item.semantic_relevance:.3f} weight={item.weight:.0f} "
            f"score={item.score:.3f} group={item.group_id} example={item.example}"
        )


def _print_decision(module: str | None, operation: str | None, status: str | None) -> None:
    print(f"selected_module={module or '-'}")
    print(f"selected_operation={operation or '-'}")
    print(f"internal_status={status or '-'}")
    print("dry_run=true writes=none paid_model=none")


if __name__ == "__main__":
    raise SystemExit(main())
