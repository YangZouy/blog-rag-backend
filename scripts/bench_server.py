#!/usr/bin/env python3
"""RAG 分阶段延迟基准测试（本地 / 服务器通用）。

设计目标（对齐诊断方案 P0-2）：
- 固定问题集（>=10 个），每问重复若干次（首冷后热）
- 直接 import 内部函数做进程内测量，绕过 HTTP 结果缓存与网络层，
  拿到 embedding / faiss / bm25 / fusion / rerank / generate 的纯净分时
- 支持模式：full（带 rerank）/ no-rerank / embed-only / generate-only
- 输出 JSON + CSV，本地与服务器用同一批问题对比

用法：
  cd /opt/blog-rag-backend
  .venv/bin/python scripts/bench_server.py --label server --mode all --repeats 3
  .venv/bin/python scripts/bench_server.py --label local  --mode full

输出：
  bench_result_<label>.json
  bench_result_<label>.csv
并在 stdout 打印汇总表。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from typing import Dict, List

# 让脚本能从项目根目录直接 import api / core
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.config import get_settings  # noqa: E402
from core.vector_store import get_vector_store  # noqa: E402
from core.bm25 import get_bm25_index, search as bm25_search  # noqa: E402
from core.embeddings import get_embeddings  # noqa: E402
from api.retriever import (  # noqa: E402
    _embed_query_cached,
    _faiss_search,
    retrieve_hybrid,
    retrieve_with_rerank,
)
from core.llm import get_gen_llm  # noqa: E402
from data.parse_hexo import DocumentChunk  # noqa: E402


# ---- 固定问题集（覆盖概念/术语/操作/闲聊类，尽量命中不同召回路径）----
DEFAULT_QUESTIONS = [
    "博客里有哪些与 AI 相关的文章？",
    "博客中LangChain相关的内容讲了什么？",
    "如何部署这个博客到服务器？",
    "TypeScript 的 unknown 和 any 有什么区别？",
    "什么是 RAG？它和普通向量检索有什么不同？",
    "BOM 头是什么？为什么会导致文件读取乱码？",
    "localStorage 和 sessionStorage 的区别？",
    "怎么用 Hexo 写一篇带标签的文章？",
    "DeepSeek 和智谱大模型在博客里是怎么被用到的？",
    "前端跨域 CORS 报错通常怎么排查？",
    "博客里有没有讲 React 性能优化的内容？",
    "什么是 RRF 融合？为什么要在 RAG 里同时用向量和 BM25？",
]

GEN_PROMPT = (
    "你是博客 AI 问答助手。请基于以下资料简洁回答用户问题（500字以内）。\n\n"
    "## 资料\n{context}\n\n## 问题\n{query}\n\n回答："
)


def _gen_answer(docs: List[DocumentChunk], query: str) -> str:
    ctx = "\n\n".join(f"### 《{d.title}》\n{d.content}" for d in docs[:5])
    return get_gen_llm().invoke(GEN_PROMPT.format(context=ctx, query=query)).content


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    return {
        "avg": round(statistics.mean(values), 1),
        "p50": round(_pct(values, 0.50), 1),
        "p95": round(_pct(values, 0.95), 1),
        "max": round(max(values), 1),
        "n": len(values),
    }


def run_bench(questions: List[str], mode: str, repeats: int, label: str) -> Dict:
    s = get_settings()
    K = s.RETRIEVAL_CANDIDATE_K
    RK = s.RERANK_CANDIDATE_K
    LIMIT = 5

    # 预热：触发 BM25 构建 + reranker 模型加载，使后续计时处于 warm 状态
    print(f"[warmup] building bm25 + loading reranker on 1 question ...", flush=True)
    retrieve_with_rerank(questions[0], top_k=LIMIT)
    print("[warmup] done", flush=True)

    samples: Dict[str, List[float]] = {
        k: [] for k in (
            "embed", "faiss", "bm25", "hybrid",
            "rerank", "retrieve_rerank", "generate",
        )
    }

    for qi, q in enumerate(questions):
        for r in range(repeats):
            # 1) embedding（不同问题 -> 真实网络；同问题第2/3次命中 lru_cache）
            t = time.perf_counter()
            qvec = list(_embed_query_cached(q))
            samples["embed"].append((time.perf_counter() - t) * 1000)

            # 2) faiss（隔离）
            t = time.perf_counter()
            _faiss_search(qvec, K)
            samples["faiss"].append((time.perf_counter() - t) * 1000)

            # 3) bm25（隔离）
            t = time.perf_counter()
            bm25_search(q, K)
            samples["bm25"].append((time.perf_counter() - t) * 1000)

            # 4) hybrid（向量+BM25+融合，embed 已缓存故几乎免费）
            t = time.perf_counter()
            hyb = retrieve_hybrid(q, top_k=LIMIT, candidate_k=RK)
            samples["hybrid"].append((time.perf_counter() - t) * 1000)

            # 5) rerank 隔离（用上面 hybrid 的候选）
            t = time.perf_counter()
            from core.rerank import rerank
            reranked = rerank(q, hyb, LIMIT)
            samples["rerank"].append((time.perf_counter() - t) * 1000)

            # 6) retrieve_with_rerank 整体（full 路径）
            if mode in ("all", "full", "no-rerank"):
                t = time.perf_counter()
                if mode == "no-rerank":
                    retrieve_hybrid(q, top_k=LIMIT, candidate_k=RK)
                else:
                    retrieve_with_rerank(q, top_k=LIMIT)
                samples["retrieve_rerank"].append((time.perf_counter() - t) * 1000)

            # 7) generate（用 rerank 结果）
            if mode in ("all", "full", "no-rerank", "generate"):
                t = time.perf_counter()
                _gen_answer(reranked if mode != "no-rerank" else hyb, q)
                samples["generate"].append((time.perf_counter() - t) * 1000)

            print(f"  q{qi+1}/{len(questions)} run{r+1} ok", flush=True)

    # 仅保留当前模式涉及的指标
    active = ["embed", "faiss", "bm25", "hybrid", "rerank"]
    if mode in ("all", "full", "no-rerank"):
        active += ["retrieve_rerank"]
    if mode in ("all", "full", "no-rerank", "generate"):
        active += ["generate"]

    aggregates = {k: _stats(samples[k]) for k in active}

    total_key = "retrieve_rerank" if mode in ("all", "full", "no-rerank") else None
    if total_key and samples.get("generate"):
        # full 端到端 = 检索整体 + 生成
        full = [a + b for a, b in zip(samples[total_key], samples["generate"])]
        aggregates["full_e2e"] = _stats(full)
        active += ["full_e2e"]

    result = {
        "label": label,
        "mode": mode,
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "questions": len(questions),
        "repeats": repeats,
        "config": {
            "rerank_candidate_k": RK,
            "rerank_max_length": s.RERANK_MAX_LENGTH,
            "retrieval_candidate_k": K,
            "warmup_on_start": s.WARMUP_ON_START,
        },
        "aggregates": aggregates,
        "raw": {k: [round(v, 1) for v in samples[k]] for k in samples if samples[k]},
    }
    return result, active


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=platform.node(), help="标识：local / server")
    ap.add_argument("--mode", default="all",
                    choices=["all", "full", "no-rerank", "embed", "generate"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--questions-file", default=None, help="每行一个问题，覆盖默认集")
    ap.add_argument("--out-dir", default=ROOT)
    args = ap.parse_args()

    if args.questions_file and os.path.isfile(args.questions_file):
        with open(args.questions_file, encoding="utf-8") as fh:
            questions = [ln.strip() for ln in fh if ln.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    # embed-only / generate-only 模式只需少量问题
    if args.mode in ("embed", "generate"):
        questions = questions[: min(len(questions), 5)]

    result, active = run_bench(questions, args.mode, args.repeats, args.label)

    base = os.path.join(args.out_dir, f"bench_result_{args.label}_{args.mode}")
    json_path = base + ".json"
    csv_path = base + ".csv"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "avg_ms", "p50_ms", "p95_ms", "max_ms", "n"])
        for k in active:
            st = result["aggregates"][k]
            w.writerow([k, st["avg"], st["p50"], st["p95"], st["max"], st["n"]])

    # stdout 汇总表
    print("\n===== Benchmark Summary =====")
    print(f"label={result['label']} mode={result['mode']} host={result['host']} "
          f"cpus={result['cpu_count']} questions={result['questions']} repeats={result['repeats']}")
    print(f"config={result['config']}")
    print(f"{'metric':<16}{'avg':>9}{'p50':>9}{'p95':>9}{'max':>9}")
    for k in active:
        st = result["aggregates"][k]
        print(f"{k:<16}{st['avg']:>9}{st['p50']:>9}{st['p95']:>9}{st['max']:>9}")
    print(f"\nJSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
