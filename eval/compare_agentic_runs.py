# -*- coding: utf-8 -*-
"""Compare two saved Agentic evaluation result files without re-running models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "multi_turn_rewrite_accuracy",
    "sub_query_coverage",
    "expected_action_accuracy",
    "no_answer_refusal_accuracy",
    "citation_support_rate",
    "average_retrieval_rounds",
    "latency_p50_sec",
    "latency_p95_sec",
    "average_estimated_token_cost",
)


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(baseline: dict, candidate: dict) -> list[dict[str, float | str | None]]:
    base = baseline.get("summary", {}).get("agentic", {})
    current = candidate.get("summary", {}).get("agentic", {})
    rows = []
    for name in METRICS:
        before, after = base.get(name), current.get(name)
        rows.append({"metric": name, "baseline": before, "agentic": after, "delta": round(after - before, 4) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and Agentic RAG evaluation results.")
    parser.add_argument("baseline")
    parser.add_argument("agentic")
    args = parser.parse_args()
    rows = compare(load(args.baseline), load(args.agentic))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
