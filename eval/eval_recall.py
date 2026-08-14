# -*- coding: utf-8 -*-
"""Bottom-layer retrieval regression for dense, hybrid/RRF, and rerank.

This suite deliberately bypasses conversation rewriting, planning, evidence
gates, remediation, and answer generation. It answers one question only:
did a retrieval-layer change move the known relevant blog slug up or down?
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.retriever import retrieve_hybrid, retrieve_with_rerank  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.embeddings import get_embeddings  # noqa: E402
from core.vector_store import get_vector_store  # noqa: E402

KS = (3, 5, 10, 12, 50)
CANDIDATE_POOL = 50
MODES = ("raw", "hybrid", "rerank")
RESULTS_DIR = ROOT / "eval" / "results"
BASELINES_DIR = ROOT / "eval" / "baselines"
QUERIES_PATH = ROOT / "eval" / "eval_queries.json"

# The historical suite has always contained 49 records. Its ids run 1..50
# with id=25 absent, which is why older documentation incorrectly called it a
# 50-query suite. Freeze the actual records instead of inventing a new case.
RETRIEVAL_REGRESSION_COUNT = 49
RETRIEVAL_REGRESSION_SHA256 = (
    "4c56ee6426d1096fa08d6267e386574ec4fcf62845760e488d5ad25050acb31e"
)

RankedSlugs = list[tuple[str, float]]
RetrieveFn = Callable[[str, int], RankedSlugs]


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(slug: str) -> str:
    return (slug or "").replace("\\", "/").strip()


def load_regression_dataset(
    path: Path = QUERIES_PATH, *, enforce_fingerprint: bool = True
) -> tuple[list[dict], dict]:
    raw = path.read_bytes()
    data = json.loads(raw)
    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{path} must contain a non-empty 'queries' list")

    required = {"id", "query", "type", "category", "expected_slugs"}
    ids: list[int] = []
    for index, query in enumerate(queries, start=1):
        missing = required - set(query)
        if missing:
            raise ValueError(f"query #{index} missing fields: {sorted(missing)}")
        if not query["query"].strip() or not query["expected_slugs"]:
            raise ValueError(f"query id={query['id']} has no query text or expected slug")
        ids.append(query["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("retrieval regression query ids must be unique")

    fingerprint = hashlib.sha256(raw).hexdigest()
    if enforce_fingerprint and (
        len(queries) != RETRIEVAL_REGRESSION_COUNT
        or fingerprint != RETRIEVAL_REGRESSION_SHA256
    ):
        raise ValueError(
            "retrieval regression dataset changed: expected the frozen historical "
            f"suite ({RETRIEVAL_REGRESSION_COUNT} records, sha256="
            f"{RETRIEVAL_REGRESSION_SHA256}), got {len(queries)} records, "
            f"sha256={fingerprint}. Create a separate dataset for new cases."
        )
    return queries, {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "record_count": len(queries),
        "sha256": fingerprint,
        "frozen": enforce_fingerprint,
    }


def _unique_slug_rank(chunks) -> RankedSlugs:
    rank: RankedSlugs = []
    seen: set[str] = set()
    for chunk in chunks:
        slug = norm(chunk.slug)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        rank.append((slug, float(chunk.score or 0.0)))
    return rank


def retrieve_raw(query: str, limit: int = CANDIDATE_POOL) -> RankedSlugs:
    qvec = get_embeddings().embed_query(query)
    chunks = []
    for chunk, score in get_vector_store().search(qvec, limit):
        chunk.score = float(score)
        chunks.append(chunk)
    return _unique_slug_rank(chunks)


def retrieve_hybrid_eval(query: str, limit: int = CANDIDATE_POOL) -> RankedSlugs:
    """Production dense + BM25 candidate ranking after RRF fusion."""
    return _unique_slug_rank(
        retrieve_hybrid(query, top_k=limit, candidate_k=limit)
    )


def retrieve_rerank_eval(query: str, limit: int = CANDIDATE_POOL) -> RankedSlugs:
    """Production hybrid/RRF candidates after cross-encoder reranking."""
    return _unique_slug_rank(
        retrieve_with_rerank(query, top_k=limit, candidate_k=limit)
    )


RETRIEVERS: dict[str, RetrieveFn] = {
    "raw": retrieve_raw,
    "hybrid": retrieve_hybrid_eval,
    "rerank": retrieve_rerank_eval,
}


def eval_one(query: dict, retrieve_fn: RetrieveFn) -> dict:
    expected = {norm(slug) for slug in query["expected_slugs"]}
    doc_rank = retrieve_fn(query["query"], CANDIDATE_POOL)
    slugs = [slug for slug, _ in doc_rank]
    recalls = {k: float(bool(set(slugs[:k]) & expected)) for k in KS}

    first_rank = next(
        (rank for rank, slug in enumerate(slugs, start=1) if slug in expected), None
    )
    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
    top1_score = doc_rank[0][1] if doc_rank else None
    relevant_scores = [score for slug, score in doc_rank if slug in expected]
    relevant_score = max(relevant_scores) if relevant_scores else None
    delta = (
        top1_score - relevant_score
        if top1_score is not None and relevant_score is not None
        else None
    )

    if relevant_score is None:
        diagnosis = "C_not_recalled"
    elif first_rank == 1:
        diagnosis = "hit_top1"
    elif delta is not None and delta < 0.05:
        diagnosis = "A_flat_scores"
    else:
        diagnosis = "B_rerank_candidate"

    return {
        "id": query["id"],
        "query": query["query"],
        "type": query["type"],
        "category": query["category"],
        "expected": sorted(expected),
        "first_rank": first_rank,
        "recall": recalls,
        "rr": reciprocal_rank,
        "top1_score": round(top1_score, 6) if top1_score is not None else None,
        "relevant_score": round(relevant_score, 6) if relevant_score is not None else None,
        "delta": round(delta, 6) if delta is not None else None,
        "diagnosis": diagnosis,
        "top5": [(slug, round(score, 6)) for slug, score in doc_rank[:5]],
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot aggregate an empty evaluation")
    count = len(rows)
    overall = {
        f"recall@{k}": round(sum(row["recall"][k] for row in rows) / count, 4)
        for k in KS
    }
    overall["MRR"] = round(sum(row["rr"] for row in rows) / count, 4)

    def group(key: str) -> dict:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row[key], []).append(row)
        return {
            name: {
                "n": len(items),
                "recall@10": round(
                    sum(item["recall"][10] for item in items) / len(items), 4
                ),
                "recall@50": round(
                    sum(item["recall"][50] for item in items) / len(items), 4
                ),
                "MRR": round(sum(item["rr"] for item in items) / len(items), 4),
            }
            for name, items in sorted(grouped.items())
        }

    deltas = [row["delta"] for row in rows if row["delta"] is not None]
    diagnoses: dict[str, int] = {}
    for row in rows:
        diagnoses[row["diagnosis"]] = diagnoses.get(row["diagnosis"], 0) + 1
    return {
        "overall": overall,
        "by_type": group("type"),
        "by_category": group("category"),
        "separation": {
            "mean_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
            "n_with_relevant_hit": len(deltas),
            "diagnoses": diagnoses,
        },
    }


def experiment_config(mode: str) -> dict:
    settings = get_settings()
    index_path = Path(settings.FAISS_INDEX_PATH)
    meta_path = Path(settings.FAISS_META_PATH)
    return {
        "retrieval_mode": mode,
        "candidate_pool": CANDIDATE_POOL,
        "rrf_k": 60,
        "embedding_model": settings.EMBED_MODEL,
        "embedding_dim": settings.EMBED_DIM,
        "index": {
            "faiss_path": str(index_path),
            "faiss_sha256": file_sha256(index_path),
            "metadata_path": str(meta_path),
            "metadata_sha256": file_sha256(meta_path),
            "built_at": (
                datetime.fromtimestamp(index_path.stat().st_mtime).astimezone().isoformat()
                if index_path.is_file()
                else None
            ),
        },
        "reranker": {
            "backend": settings.RERANK_BACKEND,
            "model": settings.RERANK_MODEL_REPO,
            "onnx_file": settings.RERANK_ONNX_FILE,
            "max_length": settings.RERANK_MAX_LENGTH,
            "configured_candidate_k": settings.RERANK_CANDIDATE_K,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def latest_previous_result(mode: str, exclude_path: Path) -> Path | None:
    candidates: list[Path] = []
    for filename in glob.glob(str(RESULTS_DIR / "retrieval_regression_*.json")):
        path = Path(filename)
        if path.resolve() == exclude_path.resolve():
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("config", {}).get("retrieval_mode") == mode:
            candidates.append(path)
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    baseline = BASELINES_DIR / f"retrieval_regression_{mode}.json"
    return baseline if baseline.is_file() else None


def metric_diff(previous: dict, current: dict) -> dict[str, dict[str, float]]:
    result = {}
    for metric in (*[f"recall@{k}" for k in KS], "MRR"):
        before = previous.get("overall", {}).get(metric)
        after = current.get("overall", {}).get(metric)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            result[metric] = {
                "previous": before,
                "current": after,
                "delta": round(after - before, 4),
            }
    return result


def run_mode(mode: str, tag: str, queries: list[dict], dataset: dict) -> tuple[dict, Path]:
    started = time.perf_counter()
    rows = [eval_one(query, RETRIEVERS[mode]) for query in queries]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(char if char.isalnum() or char in "-_" else "_" for char in tag)
    output_path = RESULTS_DIR / f"retrieval_regression_{timestamp}_{mode}_{safe_tag}.json"
    result = {
        "suite": "retrieval_regression",
        "scope": "bottom_layer_retrieval_only",
        "tag": tag,
        "timestamp": timestamp,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "dataset": dataset,
        "config": experiment_config(mode),
        "n_queries": len(queries),
        **aggregate(rows),
        "per_query": rows,
    }
    previous_path = latest_previous_result(mode, output_path)
    if previous_path:
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        result["comparison"] = {
            "previous_result": str(previous_path),
            "same_dataset": previous.get("dataset", {}).get("sha256") == dataset["sha256"],
            "metrics": metric_diff(previous, result),
        }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, output_path


def print_result(result: dict, output_path: Path) -> None:
    mode = result["config"]["retrieval_mode"]
    print(f"\n[{mode}] {result['n_queries']} queries, {result['elapsed_sec']:.3f}s")
    print("  " + "  ".join(
        f"{metric}={value:.4f}" for metric, value in result["overall"].items()
    ))
    misses = [row for row in result["per_query"] if row["recall"][10] == 0]
    print(f"  top10 misses={len(misses)}")
    for row in misses:
        print(f"    #{row['id']} {row['query']}")
    comparison = result.get("comparison")
    if comparison:
        print(f"  previous={comparison['previous_result']}")
        for metric, values in comparison["metrics"].items():
            print(
                f"    {metric}: {values['previous']:.4f} -> "
                f"{values['current']:.4f} ({values['delta']:+.4f})"
            )
    print(f"  saved={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen bottom-layer retrieval regression suite."
    )
    parser.add_argument("--tag", default="run", help="Experiment label")
    parser.add_argument(
        "--mode", choices=(*MODES, "all"), default="all",
        help="Run one retrieval stage or all controlled stages (default: all)",
    )
    parser.add_argument(
        "--allow-dataset-change", action="store_true",
        help="Explicit escape hatch for diagnostics; results are marked unfrozen",
    )
    args = parser.parse_args()

    queries, dataset = load_regression_dataset(
        enforce_fingerprint=not args.allow_dataset_change
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    modes = MODES if args.mode == "all" else (args.mode,)
    print(
        f"retrieval_regression dataset={dataset['record_count']} "
        f"sha256={dataset['sha256']}"
    )
    for mode in modes:
        result, output_path = run_mode(mode, args.tag, queries, dataset)
        print_result(result, output_path)


if __name__ == "__main__":
    main()
