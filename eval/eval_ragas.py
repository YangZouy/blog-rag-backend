# -*- coding: utf-8 -*-
"""Generate production RAG answers and evaluate groundedness with RAGAS."""
from __future__ import annotations

# 必须在任何 ragas import 之前注入 langchain-community 兼容垫片（见文件注释）。
from ._ragas_compat import _install as _install_ragas_compat  # type: ignore

import argparse
import json
import math
import os
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


def score_agentic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score trace-based behaviours without asking a judge model."""
    multi_turn = [row for row in rows if row.get("history")]
    complex_rows = [row for row in rows if row.get("expected_sub_queries")]
    refusal_rows = [row for row in rows if row.get("should_refuse")]
    action_rows = [row for row in rows if row.get("expected_action")]

    rewrite_hits = [bool(row.get("trace", {}).get("rewritten")) for row in multi_turn]
    sub_query_coverages = []
    for row in complex_rows:
        actual = [_normalise(query) for query in row.get("trace", {}).get("sub_queries", [])]
        expected = [_normalise(query) for query in row.get("expected_sub_queries", [])]
        if expected:
            sub_query_coverages.append(sum(any(target in query or query in target for query in actual) for target in expected) / len(expected))

    action_hits = [
        row.get("trace", {}).get("final_decision") == row.get("expected_action")
        for row in action_rows
    ]
    refusal_hits = [
        row.get("trace", {}).get("final_decision") == "refuse"
        for row in refusal_rows
    ]
    citation_support = []
    for row in rows:
        expected = set(row.get("expected_slugs", []))
        if expected:
            retrieved = {item.get("slug") for item in row.get("context_sources", [])}
            citation_support.append(bool(expected & retrieved))

    latencies = [float(row["latency_sec"]) for row in rows if isinstance(row.get("latency_sec"), (int, float))]
    rounds = [float(row.get("trace", {}).get("retrieval_rounds", 0)) for row in rows]
    costs = [float(row["estimated_token_cost"]) for row in rows if isinstance(row.get("estimated_token_cost"), (int, float))]
    return {
        "multi_turn_rewrite_accuracy": _mean([float(hit) for hit in rewrite_hits]),
        "sub_query_coverage": _mean(sub_query_coverages),
        "expected_action_accuracy": _mean([float(hit) for hit in action_hits]),
        "no_answer_refusal_accuracy": _mean([float(hit) for hit in refusal_hits]),
        "citation_support_rate": _mean([float(hit) for hit in citation_support]),
        "average_retrieval_rounds": _mean(rounds),
        "latency_p50_sec": _percentile(latencies, 0.5),
        "latency_p95_sec": _percentile(latencies, 0.95),
        "average_estimated_token_cost": _mean(costs),
        "counts": {"multi_turn": len(multi_turn), "complex": len(complex_rows), "refusal": len(refusal_rows), "action": len(action_rows)},
    }


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
    eligible = [row for row in rows if row["mode"] == "rag" and not row["fallback"] and row["answer"].strip() and row["contexts"]]
    if not eligible:
        return {"evaluated_rows": 0, "skipped_rows": len(rows), "metrics": [], "aggregate": {}}
    try:
        from ragas import evaluate, metrics
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    except ImportError as exc:
        raise RuntimeError("RAGAS is not installed. Run `pip install -r requirements.txt` first.") from exc

    # 无参考答案时只跑不依赖 reference 的指标（Faithfulness / AnswerRelevancy）。
    # 注意：ragas 0.4.x 的 ContextPrecision 必须有 reference，不能放进无参考集合，
    # 否则 evaluate() 会报 "requires the following additional columns ['reference']"。
    selected = [metric(metrics, "Faithfulness", "faithfulness"), metric(metrics, "ResponseRelevancy", "AnswerRelevancy", "answer_relevancy")]
    use_reference = all((row.get("reference_answer") or "").strip() for row in eligible)
    if use_reference:
        selected += [
            metric(metrics, "ContextPrecision", "ContextPrecisionWithoutReference", "context_precision"),
            metric(metrics, "LLMContextRecall", "ContextRecall", "context_recall"),
            metric(metrics, "FactualCorrectness", "AnswerCorrectness", "answer_correctness"),
        ]
    samples = [SingleTurnSample(user_input=row["query"], response=row["answer"], retrieved_contexts=row["contexts"], reference=row["reference_answer"] if use_reference else None) for row in eligible]
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
    return {"evaluated_rows": len(eligible), "skipped_rows": len(rows) - len(eligible), "metrics": names, "aggregate": aggregate, "reference_metrics_enabled": use_reference}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production RAG generation and RAGAS evaluation.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    queries = load_queries(os.path.abspath(args.dataset))
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
    }
    if not args.generate_only:
        summary["ragas"] = score_with_ragas(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings = get_settings()
    result = {"tag": args.tag, "timestamp": timestamp, "elapsed_sec": round(time.perf_counter() - started, 2), "config": {"dataset": os.path.relpath(os.path.abspath(args.dataset), ROOT), "top_k": args.top_k, "generation_context_k": settings.GENERATION_CONTEXT_K, "vector_store": "local-faiss", "generate_only": args.generate_only}, "summary": summary, "per_query": rows}
    filename = os.path.join(RESULTS_DIR, f"ragas_results_{timestamp}_{args.tag}.json")
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(safe_json(result), file, ensure_ascii=False, indent=2)
    print(json.dumps(safe_json(summary), ensure_ascii=False, indent=2))
    print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
