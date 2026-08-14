"""Evaluate post-orchestration retrieval quality on the Agentic dataset."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import get_settings
from eval.dataset_splits import load_agentic_split
from eval.eval_ragas import (
    generate_rows,
    load_queries,
    safe_json,
    score_agentic,
    score_pipeline_retrieval,
)

DEFAULT_DATASET = os.path.join(ROOT, "eval", "agentic_eval_queries.json")
RESULTS_DIR = os.path.join(ROOT, "eval", "results")


def _normalise_slug(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip()


def build_per_query_rows(
    rows: list[dict[str, Any]], ks: tuple[int, ...] = (1, 3, 5, 10)
) -> list[dict[str, Any]]:
    """Create a retrieval-focused per-query view from full generation rows."""
    output: list[dict[str, Any]] = []
    for row in rows:
        expected = [
            _normalise_slug(slug)
            for slug in row.get("expected_slugs", [])
            if _normalise_slug(slug)
        ]
        ranked: list[str] = []
        seen: set[str] = set()
        for source in row.get("context_sources", []):
            slug = _normalise_slug(source.get("slug"))
            if slug and slug not in seen:
                seen.add(slug)
                ranked.append(slug)
        first_hit_rank = next(
            (index for index, slug in enumerate(ranked, start=1) if slug in set(expected)),
            None,
        )
        retrieval = {
            "expected_slugs": expected,
            "retrieved_slugs": ranked,
            "first_hit_rank": first_hit_rank,
            "hit": bool(first_hit_rank),
        }
        for k in ks:
            matched = set(ranked[:k]) & set(expected)
            retrieval[f"hit@{k}"] = bool(matched)
            retrieval[f"coverage@{k}"] = round(len(matched) / len(expected), 4) if expected else None
            retrieval[f"all_hit@{k}"] = bool(expected) and matched == set(expected)
        output.append(
            {
                "id": row.get("id"),
                "query": row.get("query"),
                "type": row.get("type"),
                "category": row.get("category"),
                "mode": row.get("mode"),
                "expected_action": row.get("expected_action"),
                "trace": row.get("trace"),
                "retrieval": retrieval,
            }
        )
    return output


def resolve_queries(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    if args.split:
        if args.dataset != DEFAULT_DATASET:
            raise ValueError("--split cannot be combined with --dataset")
        queries, split_metadata = load_agentic_split(args.split)
        return queries, f"agentic:{args.split}", split_metadata
    return load_queries(os.path.abspath(args.dataset)), os.path.relpath(os.path.abspath(args.dataset), ROOT), None


def should_hide_per_query(args: argparse.Namespace) -> bool:
    return args.split == "final" and not args.reveal_final_per_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-orchestration retrieval evaluation on the Agentic dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["dev", "final"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="baseline")
    parser.add_argument(
        "--reveal-final-per-query",
        action="store_true",
        help="Include per-query details when evaluating the frozen final split.",
    )
    args = parser.parse_args()

    queries, dataset_label, split_metadata = resolve_queries(args)
    if args.limit is not None:
        queries = queries[: args.limit]
    if not queries:
        raise ValueError("No queries selected for evaluation.")

    started = time.perf_counter()
    rows = generate_rows(queries, args.top_k)
    per_query_hidden = should_hide_per_query(args)
    summary = {
        "generated_rows": len(rows),
        "mode_counts": {
            mode: sum(row["mode"] == mode for row in rows)
            for mode in sorted({row["mode"] for row in rows})
        },
        # Trace-level orchestration quality is evaluated alongside retrieval.
        # This keeps the dedicated entrypoint self-contained while reusing the
        # same deterministic scoring contract as eval_ragas.py.
        "agentic": score_agentic(rows),
        "post_orchestration_retrieval": score_pipeline_retrieval(rows),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings = get_settings()
    result = {
        "suite": "agentic_post_orchestration_retrieval",
        "tag": args.tag,
        "timestamp": timestamp,
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "config": {
            "dataset": dataset_label,
            "split": split_metadata,
            "top_k": args.top_k,
            "generation_context_k": settings.GENERATION_CONTEXT_K,
            "vector_store": "local-faiss",
            "per_query_hidden": per_query_hidden,
        },
        "summary": summary,
        "per_query": [] if per_query_hidden else build_per_query_rows(rows),
    }
    filename = os.path.join(
        RESULTS_DIR, f"agentic_retrieval_results_{timestamp}_{args.tag}.json"
    )
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(safe_json(result), file, ensure_ascii=False, indent=2)

    print(json.dumps(safe_json(summary), ensure_ascii=False, indent=2))
    if per_query_hidden:
        print("Per-query details hidden for final split. Re-run with --reveal-final-per-query if you need them.")
    print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
