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
    "refusal_behavior_accuracy",
    "partial_answer_behavior_accuracy",
    "out_of_scope_behavior_accuracy",
    "citation_support_rate",
    "citation_coverage_rate",
    "macro_expected_action_accuracy",
    "planning_constraint_violation_rate",
    "average_retrieval_rounds",
    "latency_p50_sec",
    "latency_p95_sec",
    "average_estimated_token_cost",
    "pipeline_retrieval.recall@1",
    "pipeline_retrieval.recall@3",
    "pipeline_retrieval.recall@5",
    "pipeline_retrieval.recall@10",
    "pipeline_retrieval.hit@1",
    "pipeline_retrieval.hit@5",
    "pipeline_retrieval.coverage@1",
    "pipeline_retrieval.coverage@5",
    "pipeline_retrieval.all_hit@5",
    "pipeline_retrieval.MRR",
    "pipeline_retrieval.slug_hit_rate",
)


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(baseline: dict, candidate: dict) -> list[dict[str, float | str | None]]:
    base = baseline.get("summary", {}).get("agentic", {})
    current = candidate.get("summary", {}).get("agentic", {})
    base_summary = baseline.get("summary", {})
    current_summary = candidate.get("summary", {})
    base_pipeline = base_summary.get("pipeline_retrieval") or base_summary.get("post_orchestration_retrieval", {})
    current_pipeline = current_summary.get("pipeline_retrieval") or current_summary.get("post_orchestration_retrieval", {})
    rows = []
    for name in METRICS:
        if name.startswith("pipeline_retrieval."):
            metric_name = name.split(".", 1)[1]
            before, after = base_pipeline.get(metric_name), current_pipeline.get(metric_name)
        else:
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
