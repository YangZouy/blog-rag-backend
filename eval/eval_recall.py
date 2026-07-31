# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import get_settings          # noqa: E402
from core.embeddings import get_embeddings     # noqa: E402
from core.qdrant_client import get_qdrant      # noqa: E402
from data.parse_hexo import DocumentChunk      # noqa: E402
from api.retriever import retrieve_hybrid, retrieve_with_rerank  # noqa: E402

KS = [3, 5, 10, 12, 50]
CANDIDATE_POOL = 50
RESULTS_DIR = os.path.join(ROOT, "eval", "results")
QUERIES_PATH = os.path.join(ROOT, "eval", "eval_queries.json")

# 纯余弦计算相似度检索
def retrieve_raw(query: str, limit: int = CANDIDATE_POOL):
    qvec = get_embeddings().embed_query(query)
    s = get_settings()
    client = get_qdrant()
    resp = client.query_points(
        collection_name=s.QDRANT_COLLECTION,
        query=qvec,
        limit=limit,
        with_payload=True,
        timeout=s.QDRANT_READ_TIMEOUT,
    )
    doc_rank = []
    seen = set()
    for p in resp.points:
        chunk_raw = (p.payload or {}).get("chunk")
        if not chunk_raw:
            continue
        c = DocumentChunk.from_payload(chunk_raw)
        slug = (c.slug or "").replace("\\", "/")
        if slug in seen:
            continue
        seen.add(slug)
        doc_rank.append((slug, float(p.score)))
    return doc_rank


def retrieve_hybrid_eval(query: str, limit: int = CANDIDATE_POOL):
    """Production vector retrieval fused with BM25 by reciprocal-rank fusion."""
    chunks = retrieve_hybrid(query, top_k=limit, candidate_k=limit)
    doc_rank = []
    seen = set()
    for chunk in chunks:
        slug = (chunk.slug or "").replace("\\", "/")
        if slug in seen:
            continue
        seen.add(slug)
        doc_rank.append((slug, float(chunk.score or 0.0)))
    return doc_rank

def retrieve_rerank_eval(query: str, limit: int = CANDIDATE_POOL):
    """Hybrid retrieval followed by cross-encoder reranking."""
    chunks = retrieve_with_rerank(query, top_k=limit, candidate_k=limit)
    doc_rank = []
    seen = set()
    for chunk in chunks:
        slug = (chunk.slug or "").replace("\\", "/")
        if slug in seen:
            continue
        seen.add(slug)
        doc_rank.append((slug, float(chunk.score or 0.0)))
    return doc_rank

# 路径归一
def norm(slug: str) -> str:
    return (slug or "").replace("\\", "/").strip()

# 计算单条query的评估数值：recall@k MRR(首个命中排第几) Δ分离度（相关文档余弦分数 判断扁平/rerank/召回）
def eval_one(q, retrieve_fn):
    expected = {norm(s) for s in q["expected_slugs"]}
    doc_rank = retrieve_fn(q["query"])
    slugs = [norm(s) for s, _ in doc_rank]

    # recall
    recalls = {}
    for k in KS:
        recalls[k] = 1.0 if (set(slugs[:k]) & expected) else 0.0

    # MRR
    rr = 0.0
    first_rank = None
    for i, s in enumerate(slugs, start=1):
        if s in expected:
            rr = 1.0 / i
            first_rank = i
            break

    # Δ
    top1_cos = doc_rank[0][1] if doc_rank else None
    rel_cos = None
    for s, cos in doc_rank:
        if s in expected:
            rel_cos = cos if rel_cos is None else max(rel_cos, cos)
    delta = (top1_cos - rel_cos) if (top1_cos is not None and rel_cos is not None) else None

    if rel_cos is None:
        cls = "C_未召回"
    elif first_rank == 1:
        cls = "命中top1"
    elif delta is not None and delta < 0.05:
        cls = "A_扁平"
    else:
        cls = "B_可rerank"

    return {
        "id": q["id"],
        "query": q["query"],
        "type": q["type"],
        "category": q["category"],
        "expected": sorted(expected),
        "first_rank": first_rank,
        "recall": recalls,
        "rr": rr,
        "top1_cos": round(top1_cos, 4) if top1_cos is not None else None,
        "rel_cos": round(rel_cos, 4) if rel_cos is not None else None,
        "delta": round(delta, 4) if delta is not None else None,
        "class": cls,
        "top5": [(s, round(c, 4)) for s, c in doc_rank[:5]],
    }

# 汇总诊断
def aggregate(rows):
    n = len(rows)
    overall = {f"recall@{k}": round(sum(r["recall"][k] for r in rows) / n, 4) for k in KS}
    overall["MRR"] = round(sum(r["rr"] for r in rows) / n, 4)

    def group(key):
        g = {}
        for r in rows:
            g.setdefault(r[key], []).append(r)
        out = {}
        for name, rs in sorted(g.items()):
            m = len(rs)
            out[name] = {
                "n": m,
                "recall@10": round(sum(x["recall"][10] for x in rs) / m, 3),
                "recall@50": round(sum(x["recall"][50] for x in rs) / m, 3),
                "MRR": round(sum(x["rr"] for x in rs) / m, 3),
            }
        return out

    deltas = [r["delta"] for r in rows if r["delta"] is not None]
    classes = {}
    for r in rows:
        classes[r["class"]] = classes.get(r["class"], 0) + 1
    sep = {
        "mean_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "n_with_rel": len(deltas),
        "classes": classes,
    }
    return {
        "overall": overall,
        "by_type": group("type"),
        "by_category": group("category"),
        "separation": sep,
    }

# 找旧文件对比
def latest_prev_result(exclude_path):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "eval_results_*.json")))
    files = [f for f in files if os.path.abspath(f) != os.path.abspath(exclude_path)]
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tag", default="run", help="本次运行标签，如 baseline / after_bm25"
    )
    ap.add_argument(
        "--mode", default="raw", choices=["raw", "hybrid", "rerank"],
        help="raw=纯向量; hybrid=prod+BM25; rerank=hybrid+精排",
    )
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(QUERIES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]

    s = get_settings()
    retrieve_fn = {
        "raw": retrieve_raw,
        "hybrid": retrieve_hybrid_eval,
        "rerank": retrieve_rerank_eval,
    }[args.mode]
    t0 = time.time()
    rows = []
    for q in queries:
        rows.append(eval_one(q, retrieve_fn))
    elapsed = time.time() - t0

    agg = aggregate(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "tag": args.tag,
        "timestamp": ts,
        "elapsed_sec": round(elapsed, 1),
        "config": {
            "model": getattr(s, "EMBED_MODEL", "?"),
            "dim": getattr(s, "EMBED_DIM", "?"),
            "collection": s.QDRANT_COLLECTION,
            "candidate_pool": CANDIDATE_POOL,
            "retrieve": args.mode,
        },
        "n_queries": len(queries),
        **agg,
        "per_query": rows,
    }
    out_path = os.path.join(RESULTS_DIR, f"eval_results_{ts}_{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- 打印 ----
    print(f"\n{'='*60}")
    print(f"  评估结果 [{args.tag}]  模型={out['config']['model']} 维度={out['config']['dim']}")
    print(f"  {len(queries)} 条 query  耗时 {elapsed:.1f}s")
    print(f"{'='*60}")
    print("总体指标:")
    for k in KS:
        print(f"  recall@{k:<3} = {agg['overall'][f'recall@{k}']:.3f}")
    print(f"  MRR      = {agg['overall']['MRR']:.3f}")

    print("\n分 query 类型:")
    print(f"  {'type':<10}{'n':>4}{'R@10':>8}{'R@50':>8}{'MRR':>8}")
    for name, m in agg["by_type"].items():
        print(f"  {name:<10}{m['n']:>4}{m['recall@10']:>8.3f}{m['recall@50']:>8.3f}{m['MRR']:>8.3f}")

    print("\n分内容类别:")
    print(f"  {'category':<14}{'n':>4}{'R@10':>8}{'R@50':>8}{'MRR':>8}")
    for name, m in agg["by_category"].items():
        print(f"  {name:<14}{m['n']:>4}{m['recall@10']:>8.3f}{m['recall@50']:>8.3f}{m['MRR']:>8.3f}")

    sep = agg["separation"]
    print("\n分离度诊断:")
    print(f"  平均 Δ(top1 - 相关文档) = {sep['mean_delta']}  (有相关命中 {sep['n_with_rel']}/{len(queries)})")
    print(f"  分类: {sep['classes']}")
    print("  判读: A_扁平→换模型/混合检索  B_可rerank→精排  C_未召回→清洗/混合检索")

    fails = [r for r in rows if r["recall"][10] == 0.0]
    if fails:
        print(f"\n未进 top10 的 query ({len(fails)}):")
        for r in fails:
            print(f"  #{r['id']:<3} [{r['class']:<9}] {r['query']}  (rel_cos={r['rel_cos']}, top1={r['top1_cos']})")

    # ---- diff 上一次 ----
    prev = latest_prev_result(out_path)
    if prev:
        with open(prev, encoding="utf-8") as f:
            pd = json.load(f)
        print(f"\n{'='*60}")
        print(f"  对比上次: {os.path.basename(prev)} [{pd.get('tag')}]")
        print(f"{'='*60}")
        for k in KS:
            key = f"recall@{k}"
            cur = agg["overall"][key]
            old = pd.get("overall", {}).get(key, 0)
            diff = cur - old
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
            print(f"  {key:<10} {old:.3f} → {cur:.3f}  ({arrow}{diff:+.3f})")
        cur_mrr = agg["overall"]["MRR"]
        old_mrr = pd.get("overall", {}).get("MRR", 0)
        diff_mrr = cur_mrr - old_mrr
        arrow = "↑" if diff_mrr > 0 else ("↓" if diff_mrr < 0 else "=")
        print(f"  {'MRR':<10} {old_mrr:.3f} → {cur_mrr:.3f}  ({arrow}{diff_mrr:+.3f})")

    print(f"\n结果已保存: {out_path}")

if __name__ == "__main__":
    main()
