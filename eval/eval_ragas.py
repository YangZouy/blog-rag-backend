# -*- coding: utf-8 -*-
"""Generate production RAG answers and evaluate groundedness with RAGAS."""
from __future__ import annotations

# 必须在任何 ragas import 之前注入 langchain-community 兼容垫片（见文件注释）。
from ._ragas_compat import _install as _install_ragas_compat  # type: ignore

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.models import ConversationTurn
from api.rag_graph import run_rag_with_trace
from core.config import get_settings
from core.embeddings import get_embeddings
from core.llm import get_gen_llm
from eval.dataset_splits import load_agentic_split

DEFAULT_DATASET = os.path.join(ROOT, "eval", "eval_queries.json")
RESULTS_DIR = os.path.join(ROOT, "eval", "results")


def safe_json(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    return value


def load_queries(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    rows = data.get("queries", data) if isinstance(data, dict) else data
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not row.get("query") for row in rows):
        raise ValueError("Dataset must be a list of objects with a non-empty 'query'.")
    return rows


def generate_rows(queries: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    output = []
    for index, item in enumerate(queries, start=1):
        started = time.perf_counter()
        print(f"[{index}/{len(queries)}] generating: {item['query']}", flush=True)
        try:
            history = [ConversationTurn.model_validate(turn) for turn in item.get("history", [])]
            response, docs = run_rag_with_trace(item["query"], top_k=top_k, history=history)
            row = {
                "id": item.get("id"), "query": item["query"], "type": item.get("type"),
                "category": item.get("category"), "expected_slugs": item.get("expected_slugs", []),
                "reference_answer": item.get("reference_answer"), "answer": response.answer,
                "mode": response.mode, "fallback": response.fallback,
                "expected_action": item.get("expected_action"), "expected_sub_queries": item.get("expected_sub_queries", []),
                "should_refuse": bool(item.get("should_refuse", False)), "required_facts": item.get("required_facts", []),
                "history": item.get("history", []), "evidence_status": response.evidence_status,
                "missing_aspects": response.missing_aspects, "trace": response.trace.model_dump() if response.trace else None,
                "contexts": [doc.content for doc in docs if doc.content],
                "context_sources": [{"slug": doc.slug, "title": doc.title, "url": doc.url, "score": doc.score} for doc in docs],
                "citations": [citation.model_dump() for citation in response.citations],
            }
        except Exception as exc:
            row = {
                "id": item.get("id"), "query": item["query"], "type": item.get("type"),
                "category": item.get("category"), "expected_slugs": item.get("expected_slugs", []),
                "reference_answer": item.get("reference_answer"), "answer": "", "mode": "error", "fallback": True,
                "expected_action": item.get("expected_action"), "expected_sub_queries": item.get("expected_sub_queries", []),
                "should_refuse": bool(item.get("should_refuse", False)), "required_facts": item.get("required_facts", []),
                "history": item.get("history", []), "evidence_status": None, "missing_aspects": [], "trace": None,
                "contexts": [], "context_sources": [], "citations": [], "error": str(exc),
            }
        row["latency_sec"] = round(time.perf_counter() - started, 3)
        # Providers expose usage inconsistently across OpenAI-compatible APIs;
        # use a transparent character-based proxy until provider usage is available.
        row["estimated_token_cost"] = round((len(item["query"]) + len(row.get("answer", ""))) / 2, 1)
        output.append(row)
    return output


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 3)


def _normalise(text: str) -> str:
    return "".join((text or "").lower().split())


_AGENTIC_SUBSETS = {
    "multi_turn_rewrite_accuracy": lambda r: bool(r.get("history")),
    "sub_query_coverage": lambda r: bool(r.get("expected_sub_queries")),
    "expected_action_accuracy": lambda r: bool(r.get("expected_action")),
    "no_answer_refusal_accuracy": lambda r: bool(r.get("should_refuse")),
    "refusal_behavior_accuracy": lambda r: bool(r.get("should_refuse")),
    "citation_support_rate": lambda r: bool(r.get("expected_slugs")),
}


def _agentic_value(sub_rows: list[dict[str, Any]], name: str) -> float | None:
    """Compute one agentic metric from an ALREADY-filtered row subset."""
    if name == "multi_turn_rewrite_accuracy":
        return _mean([float(bool(r.get("trace", {}).get("rewritten"))) for r in sub_rows])
    if name == "sub_query_coverage":
        covs = []
        for r in sub_rows:
            actual = [_normalise(q) for q in r.get("trace", {}).get("sub_queries", [])]
            expected = [_normalise(q) for q in r.get("expected_sub_queries", [])]
            if expected:
                covs.append(
                    sum(any(t in q or q in t for q in actual) for t in expected) / len(expected)
                )
        return _mean(covs)
    if name == "expected_action_accuracy":
        hits = [bool(r.get("trace", {}).get("final_decision") == r.get("expected_action")) for r in sub_rows]
        return _mean([float(h) for h in hits])
    if name == "no_answer_refusal_accuracy":
        hits = [bool(r.get("trace", {}).get("final_decision") == "refuse") for r in sub_rows]
        return _mean([float(h) for h in hits])
    if name == "refusal_behavior_accuracy":
        hits = [
            bool(
                (r.get("trace") or {}).get("final_decision") == "refuse"
                and not r.get("citations")
                and not (r.get("answer") or "").strip()
            )
            for r in sub_rows
        ]
        return _mean([float(h) for h in hits])
    if name == "citation_support_rate":
        supp = [
            bool(set(r.get("expected_slugs", [])) & {it.get("slug") for it in r.get("context_sources", [])})
            for r in sub_rows
        ]
        return _mean([float(s) for s in supp])
    return None


def score_agentic(rows: list[dict[str, Any]], n_boot: int = 1000) -> dict[str, Any]:
    """Score trace-based behaviours without asking a judge model.

    Each behavioural metric is computed over its own subset and reported with a
    bootstrap 95% CI, because several subsets are tiny (e.g. 1-8 cases) and a raw
    point estimate there is mostly sampling noise.
    """
    metric_names = (
        "multi_turn_rewrite_accuracy",
        "sub_query_coverage",
        "expected_action_accuracy",
        "no_answer_refusal_accuracy",
        "citation_support_rate",
        "refusal_behavior_accuracy",
    )
    metrics: dict[str, Any] = {}
    ci95: dict[str, Any] = {}
    for name in metric_names:
        pred = _AGENTIC_SUBSETS[name]
        sub = [r for r in rows if pred(r)]
        metrics[name] = _agentic_value(sub, name)
        if len(sub) >= 2:
            rng = random.Random(0)
            vals = []
            for _ in range(n_boot):
                sample = [sub[rng.randrange(len(sub))] for _ in range(len(sub))]
                v = _agentic_value(sample, name)
                if isinstance(v, (int, float)):
                    vals.append(v)
            ci = _ci_bounds(vals)
            if ci:
                ci95[name] = ci

    multi_turn = [r for r in rows if r.get("history")]
    complex_rows = [r for r in rows if r.get("expected_sub_queries")]
    refusal_rows = [r for r in rows if r.get("should_refuse")]
    action_rows = [r for r in rows if r.get("expected_action")]
    latencies = [float(r["latency_sec"]) for r in rows if isinstance(r.get("latency_sec"), (int, float))]
    rounds = [float(r.get("trace", {}).get("retrieval_rounds", 0)) for r in rows]
    costs = [float(r["estimated_token_cost"]) for r in rows if isinstance(r.get("estimated_token_cost"), (int, float))]
    return {
        **metrics,
        "average_retrieval_rounds": _mean(rounds),
        "latency_p50_sec": _percentile(latencies, 0.5),
        "latency_p95_sec": _percentile(latencies, 0.95),
        "average_estimated_token_cost": _mean(costs),
        "ci95": ci95,
        "counts": {
            "multi_turn": len(multi_turn),
            "complex": len(complex_rows),
            "refusal": len(refusal_rows),
            "action": len(action_rows),
        },
    }


def _retrieval_metrics(labelled: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, Any] | None:
    """Compute recall@k / MRR / slug_hit_rate for a list of (labelled) rows."""
    if not labelled:
        return None
    recalls = {k: [] for k in ks}
    reciprocal_ranks: list[float] = []
    slug_hits: list[bool] = []
    for row in labelled:
        expected = {
            str(slug).replace("\\", "/").strip()
            for slug in row.get("expected_slugs", [])
        }
        ranked, seen = [], set()
        for source in row.get("context_sources", []):
            slug = str(source.get("slug") or "").replace("\\", "/").strip()
            if slug and slug not in seen:
                seen.add(slug)
                ranked.append(slug)
        first_rank = next(
            (index for index, slug in enumerate(ranked, start=1) if slug in expected),
            None,
        )
        slug_hits.append(bool(first_rank))
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for k in ks:
            recalls[k].append(float(bool(set(ranked[:k]) & expected)))
    return {
        **{f"recall@{k}": round(sum(v) / len(v), 4) for k, v in recalls.items()},
        "MRR": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        "slug_hit_rate": round(sum(slug_hits) / len(slug_hits), 4),
        "n": len(labelled),
    }


def _ci_bounds(values: list[float]) -> list[float] | None:
    """95% CI via sorted percentile indices (nearest-rank)."""
    if len(values) < 2:
        return None
    ordered = sorted(values)
    n = len(ordered)
    lo = ordered[max(0, int(math.ceil(0.025 * n)) - 1)]
    hi = ordered[min(n - 1, int(math.floor(0.975 * n)))]
    return [round(lo, 4), round(hi, 4)]


def score_pipeline_retrieval(
    rows: list[dict[str, Any]], ks: tuple[int, ...] = (1, 3, 5, 10), n_boot: int = 1000
) -> dict[str, Any]:
    """Score final generation context after the full Agentic retrieval pipeline.

    Returns aggregate metrics, a per-`type` breakdown, and bootstrap 95% CIs so a
    small dev/final gap can be judged against sampling noise rather than eyeballed.
    """
    labelled = [row for row in rows if row.get("expected_slugs")]
    if not labelled:
        return {
            **{f"recall@{k}": None for k in ks},
            "MRR": None,
            "slug_hit_rate": None,
            "evaluated_rows": 0,
            "by_type": {},
            "ci95": {},
        }
    point = _retrieval_metrics(labelled, ks)
    by_type: dict[str, Any] = {}
    for t in sorted({row.get("type") for row in labelled}):
        sub = [row for row in labelled if row.get("type") == t]
        metric = _retrieval_metrics(sub, ks)
        if metric:
            by_type[t] = metric
    ci95: dict[str, Any] = {}
    if len(labelled) >= 2:
        rng = random.Random(0)
        boots: list[dict[str, Any] | None] = []
        for _ in range(n_boot):
            sample = [labelled[rng.randrange(len(labelled))] for _ in range(len(labelled))]
            boots.append(_retrieval_metrics(sample, ks))
        for key in ("recall@1", "recall@5", "MRR"):
            vals = [b[key] for b in boots if b and isinstance(b.get(key), (int, float))]
            ci = _ci_bounds(vals)
            if ci:
                ci95[key] = ci
    return {**point, "evaluated_rows": len(labelled), "by_type": by_type, "ci95": ci95}


def metric(metrics_module: Any, *names: str) -> Any:
    for name in names:
        candidate = getattr(metrics_module, name, None)
        if candidate is not None:
            return candidate() if isinstance(candidate, type) else candidate
    raise RuntimeError(f"Installed ragas does not expose any of: {', '.join(names)}")


def ragas_clients() -> tuple[Any, Any]:
    llm, embeddings = get_gen_llm(), get_embeddings()
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        # bypass_n=True: 你的 GEN_BASE_URL（智谱/DeepSeek 兼容端点）只支持 n=1，
        # 而 ragas 的 Faithfulness/AnswerRelevancy 默认会以 n>1 发多候选采样请求，
        # 端点返回 400 "Invalid n value"。开启后 ragas 改为发 n 个独立请求（每个 n=1），
        # 兼容该端点，分数等价。
        return LangchainLLMWrapper(llm, bypass_n=True), LangchainEmbeddingsWrapper(embeddings)
    except ImportError:
        return llm, embeddings


def score_with_ragas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row["mode"] == "rag"
        and not row["fallback"]
        and row["answer"].strip()
        and row["contexts"]
        and (row.get("reference_answer") or "").strip()
        and not row.get("should_refuse")
    ]
    if not eligible:
        return {"evaluated_rows": 0, "skipped_rows": len(rows), "metrics": [], "aggregate": {}}
    try:
        from ragas import evaluate, metrics
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    except ImportError as exc:
        raise RuntimeError("RAGAS is not installed. Run `pip install -r requirements.txt` first.") from exc

    selected = [metric(metrics, "Faithfulness", "faithfulness"), metric(metrics, "ResponseRelevancy", "AnswerRelevancy", "answer_relevancy")]
    selected += [
        metric(metrics, "ContextPrecision", "ContextPrecisionWithoutReference", "context_precision"),
        metric(metrics, "LLMContextRecall", "ContextRecall", "context_recall"),
        metric(metrics, "FactualCorrectness", "AnswerCorrectness", "answer_correctness"),
    ]
    samples = [SingleTurnSample(user_input=row["query"], response=row["answer"], retrieved_contexts=row["contexts"], reference=row["reference_answer"]) for row in eligible]
    llm, embeddings = ragas_clients()
    results = evaluate(EvaluationDataset(samples=samples), metrics=selected, llm=llm, embeddings=embeddings)
    scores = [safe_json(score) for score in results.to_pandas().to_dict(orient="records")]
    for row, score in zip(eligible, scores):
        row["ragas"] = score
    names = [getattr(item, "name", item.__class__.__name__) for item in selected]
    aggregate = {}
    for name in names:
        numeric = [float(score[name]) for score in scores if isinstance(score.get(name), (int, float)) and not math.isnan(score[name])]
        if numeric:
            aggregate[name] = round(sum(numeric) / len(numeric), 4)
    return {"evaluated_rows": len(eligible), "skipped_rows": len(rows) - len(eligible), "metrics": names, "aggregate": aggregate, "reference_metrics_enabled": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production RAG generation and RAGAS evaluation.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--split", choices=["dev", "final"],
        help="Use a frozen subset of eval/agentic_eval_queries.json; cannot be combined with --dataset.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if args.split:
        if args.dataset != DEFAULT_DATASET:
            parser.error("--split cannot be combined with --dataset")
        queries, split_metadata = load_agentic_split(args.split)
        dataset_label = f"agentic:{args.split}"
    else:
        queries = load_queries(os.path.abspath(args.dataset))
        split_metadata = None
        dataset_label = os.path.relpath(os.path.abspath(args.dataset), ROOT)
    if args.limit is not None:
        queries = queries[:args.limit]
    if not queries:
        raise ValueError("No queries selected for evaluation.")
    started = time.perf_counter()
    rows = generate_rows(queries, args.top_k)
    summary: dict[str, Any] = {
        "generated_rows": len(rows),
        "mode_counts": {mode: sum(row["mode"] == mode for row in rows) for mode in sorted({row["mode"] for row in rows})},
        "agentic": score_agentic(rows),
        "pipeline_retrieval": score_pipeline_retrieval(rows),
    }
    if not args.generate_only:
        summary["ragas"] = score_with_ragas(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings = get_settings()
    result = {"tag": args.tag, "timestamp": timestamp, "elapsed_sec": round(time.perf_counter() - started, 2), "config": {"dataset": dataset_label, "split": split_metadata, "top_k": args.top_k, "generation_context_k": settings.GENERATION_CONTEXT_K, "vector_store": "local-faiss", "generate_only": args.generate_only}, "summary": summary, "per_query": rows}
    filename = os.path.join(RESULTS_DIR, f"ragas_results_{timestamp}_{args.tag}.json")
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(safe_json(result), file, ensure_ascii=False, indent=2)
    print(json.dumps(safe_json(summary), ensure_ascii=False, indent=2))
    print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
