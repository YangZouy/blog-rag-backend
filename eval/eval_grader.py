"""校准并评估 CRAG 的「分数 -> 3 类」评分器与 web 触发决策。

为什么能评：eval/eval_queries.json 每条带 expected_slugs，可作为代理真值——
  - 文档级：doc.slug ∈ expected_slugs 视为 CORRECT 代理，否则 INCORRECT 代理
    （AMBIGUOUS 无真值，故只在"非模糊"预测上算 doc 级准确率）
  - 查询级：need_web 代理 = (top5 中无 expected slug) OR 查询命中时间敏感词
    （静态博客天然答不出时效问题，应当 web 兜底）

本脚本做三件事：
  1) 打印 reranker 分数在 CORRECT/INCORRECT 两类的分布，供人工 sanity check；
  2) 网格扫描 (CORRECT 阈值, INCORRECT 阈值)，以 doc 级准确率为主要目标挑工作点，
     并打印该点的预测分布（CORRECT/INCORRECT/AMBIGUOUS 各多少）；
  3) WEB_PROBES：用一批"知识缺口/时效"对抗样本，验证 web 门控真的能 fire
     （eval 集因 R@5≈1.0 几乎无 web 正例，无法靠它校准，必须靠对抗样本）。

注意：doc 级代理基于 slug 归属，是"弱标签"（同 slug 的低质 chunk 也被标 CORRECT），
绝对 doc_acc 仅供参考，重点是阈值带来的相对趋势 + 预测分布是否合理。

用法：
  .venv/Scripts/python.exe eval/eval_grader.py --top-k 12 --dump
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# 允许从项目根目录直接运行（复用现有 eval_recall.py 的调用约定）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.crag import classify_documents, decide_web_search, is_time_sensitive, GradeLabel  # noqa: E402

EVAL_PATH = Path(__file__).parent / "eval_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"

# 对抗样本：eval 集里几乎没有"知识库答不出"的正例，用这些验证 web 门控能否 fire。
# 每条应当触发 web（要么 KB 无此内容，要么有时效缺口）。
WEB_PROBES = [
    "2026 年最新的 React 19 有哪些 breaking changes",
    "今天英伟达的股价是多少",
    "Python 4.0 什么时候发布，有哪些新特性",
    "GPT-5 现在的 API 价格是多少",
    "最新的 Vue 3.5 发布了哪些新功能",
]


def load_queries() -> list:
    raw = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("queries", [])
    return raw

"""
分数转换为三种标签
"""
def _label_proxy(score: float | None, ct: float, it: float) -> GradeLabel:
    if score is None:
        return GradeLabel.AMBIGUOUS
    if score >= ct:
        return GradeLabel.CORRECT
    if score <= it:
        return GradeLabel.INCORRECT
    return GradeLabel.AMBIGUOUS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=12, help="每次检索取多少 rerank 候选")
    ap.add_argument("--dump", action="store_true", help="把原始分数写盘")
    args = ap.parse_args()

    queries = load_queries()
    print(f"[crag-eval] 加载 {len(queries)} 条查询，top_k={args.top_k}")

    # 先各自检索一次，缓存 docs + 代理标签 + 真值 need_web
    cached = []
    correct_scores: list[float] = []
    incorrect_scores: list[float] = []
    for q in queries:
        query = q["query"]
        expected = set(q.get("expected_slugs") or [])
        try:
            from api.retriever import retrieve_with_rerank

            docs = retrieve_with_rerank(query, top_k=args.top_k)
        except Exception as exc:  # 检索失败则跳过该条
            print(f"  ! 检索失败，跳过: {query!r} ({exc})")
            continue
        labels = []
        for d in docs:
            is_correct = d.slug in expected
            (correct_scores if is_correct else incorrect_scores).append(d.score or 0.0)
            labels.append((d, is_correct))
        top5_slugs = {d.slug for d in docs[:5]}
        # 代理真值：top5 无 expected slug，或查询时间敏感 -> 应当 web
        # (expected & top5_slugs)表示是否有交集
        need_web = (not (expected & top5_slugs)) or is_time_sensitive(query)
        cached.append({"q": q, "docs": docs, "labels": labels, "need_web": need_web})

    if not cached:
        print("[crag-eval] 无可用查询结果，退出")
        return

    # 确认正确分数与错误分数二者是否有可分性：若正确文档的分数整体高于错误文档，说明可用阈值切分
    # 约定_开头表示模块内部辅助函数：内部统计函数
    def _stat(name: str, xs: list[float]) -> None:
        if not xs:
            print(f"  {name}: (空)")
            return
        # 返回新列表
        xs_sorted = sorted(xs)
        print(
            # 平均值
            f"  {name}: n={len(xs)} mean={statistics.mean(xs):.3f} "
            # 中位数
            f"median={statistics.median(xs):.3f} "
            f"p10={xs_sorted[len(xs)//10]:.3f} p90={xs_sorted[min(len(xs)-1, len(xs)*9//10)]:.3f}"
        )

    print("[crag-eval] reranker 分数分布（CORRECT/INCORRECT 代理）:")
    _stat("CORRECT ", correct_scores)
    _stat("INCORRECT", incorrect_scores)

    # 网格扫描：优化目标 = useful_acc = (1 - 模糊带占比) * doc_acc
    #   单纯最大化 doc_acc 会被"把一切推入模糊带"骗到（退化解：it 极低时 ambiguous≈80%，
    #   精炼形同虚设）。useful_acc 同时奖励"解析更多文档"和"解析准确"，自动避开退化解。
    #   候选约束：模糊带 <=15% 且 保留召回 keepR >=0.65（避免误丢过多相关文档）。
    correct_cands = [i / 10 for i in range(-10, 21)]   # -1.0 .. 2.0，步长 0.1
    incorrect_cands = [i / 10 for i in range(-5, 11)]   # -0.5 .. 1.0，步长 0.1
    results = []
    for ct in correct_cands:
        for it in incorrect_cands:
            if it >= ct:
                continue
            tp = fp = fn = tn = 0
            doc_total = doc_ok = 0
            drop_tot = drop_ok = 0      # 被判 INCORRECT 中真 INCORRECT 的比例（精炼精度）
            keep_tot = keep_ok = 0      # 真 CORRECT 中未被误丢的比例（保留召回）
            pred_counts = defaultdict(int)
            for item in cached:
                classified = [
                    (d, _label_proxy(d.score, ct, it)) for d, _ in item["labels"]
                ]
                for _, lbl in classified:
                    pred_counts[lbl.value] += 1
                pred = decide_web_search(item["q"]["query"], classified)
                if item["need_web"] and pred:
                    tp += 1
                elif item["need_web"] and not pred:
                    fn += 1
                elif (not item["need_web"]) and pred:
                    fp += 1
                else:
                    tn += 1
                # doc 级指标（只看非 AMBIGUOUS 预测）
                for (d, is_correct), (_, lbl) in zip(item["labels"], classified):
                    if lbl == GradeLabel.AMBIGUOUS:
                        continue
                    doc_total += 1
                    if (lbl == GradeLabel.CORRECT) == is_correct:
                        doc_ok += 1
                    if lbl == GradeLabel.INCORRECT:
                        drop_tot += 1
                        if not is_correct:
                            drop_ok += 1
                    if is_correct:
                        keep_tot += 1
                        if lbl != GradeLabel.INCORRECT:
                            keep_ok += 1
            total_docs = sum(pred_counts.values()) or 1
            amb_frac = pred_counts.get("ambiguous", 0) / total_docs
            denom = tp + fp + fn + tn
            web_acc = (tp + tn) / denom if denom else 0.0
            web_prec = tp / (tp + fp) if (tp + fp) else 0.0
            web_rec = tp / (tp + fn) if (tp + fn) else 0.0
            web_f1 = (2 * web_prec * web_rec / (web_prec + web_rec)) if (web_prec + web_rec) else 0.0
            doc_acc = doc_ok / doc_total if doc_total else 0.0
            drop_prec = drop_ok / drop_tot if drop_tot else 0.0
            keep_rec = keep_ok / keep_tot if keep_tot else 0.0
            useful = (1 - amb_frac) * doc_acc
            results.append({
                "correct_t": ct, "incorrect_t": it,
                "web_acc": web_acc, "web_prec": web_prec, "web_rec": web_rec, "web_f1": web_f1,
                "doc_acc": doc_acc, "amb_frac": amb_frac,
                "drop_prec": drop_prec, "keep_rec": keep_rec, "useful": useful,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "pred_counts": dict(pred_counts),
            })

    # 候选约束：模糊带 <=15% 且 保留召回 >=0.65；否则视为退化解（精炼失效/误丢过多）
    constrained = [r for r in results if r["amb_frac"] <= 0.15 and r["keep_rec"] >= 0.65]
    pool = constrained if constrained else results
    pool.sort(key=lambda r: r["useful"], reverse=True)
    results.sort(key=lambda r: r["useful"], reverse=True)
    print("\n[crag-eval] 分类工作点扫描 Top8（按 useful_acc=(1-模糊带)*doc_acc）:")
    print(f"  {'ct':>5} {'it':>5} {'docA':>5} {'amb%':>5} {'dropP':>6} {'keepR':>6}   conf(tp/fp/fn/tn)")
    for r in results[:8]:
        print(f"  {r['correct_t']:5.1f} {r['incorrect_t']:5.1f} {r['doc_acc']:5.3f} "
              f"{r['amb_frac']*100:5.1f} {r['drop_prec']:6.3f} {r['keep_rec']:6.3f}   "
              f"{r['tp']}/{r['fp']}/{r['fn']}/{r['tn']}")

    best = pool[0]
    pc = best["pred_counts"]
    n_docs = sum(pc.values()) or 1
    print(f"\n[crag-eval] 建议分类工作点（约束 amb%<=15 且 keepR>=0.65）: "
          f"CORRECT_THRESHOLD={best['correct_t']:.1f}, INCORRECT_THRESHOLD={best['incorrect_t']:.1f}")
    print(f"           doc_acc={best['doc_acc']:.3f}  dropP={best['drop_prec']:.3f}  "
          f"keepR={best['keep_rec']:.3f}")
    print(f"           预测分布: correct={pc.get('correct',0)} / incorrect={pc.get('incorrect',0)} "
          f"/ ambiguous={pc.get('ambiguous',0)} "
          f"(模糊带 {pc.get('ambiguous',0)/n_docs*100:.1f}% — Phase D LLM 仅裁决此带)")

    # ------------------------------------------------------------------
    # WEB_PROBES：用对抗样本验证门控真的会 fire（eval 集无法验证）
    # ------------------------------------------------------------------
    print("\n[crag-eval] WEB_PROBES（对抗样本，应当触发 web）:")
    probe_rows = []
    for pq in WEB_PROBES:
        try:
            from api.retriever import retrieve_with_rerank

            docs = retrieve_with_rerank(pq, top_k=args.top_k)
            classified = classify_documents(docs)
            fired = decide_web_search(pq, classified)
            n_inc = sum(1 for _, l in classified[:5] if l == GradeLabel.INCORRECT)
            max_s = max((d.score or float("-inf")) for d, _ in classified[:5])
            reason = "FIRED" if fired else "not-fired"
            probe_rows.append((pq, fired, n_inc, max_s))
            print(f"  [{reason:9}] n_incorrect@5={n_inc} max_score@5={max_s:+.3f}  {pq}")
        except Exception as exc:
            print(f"  [error] {pq!r} ({exc})")
    n_fired = sum(1 for _, f, _, _ in probe_rows if f)
    print(f"  -> {n_fired}/{len(probe_rows)} 触发（预期接近 {len(probe_rows)}；"
          f"若远低于，说明门控仍偏紧，调高 INCORRECT 阈值或调低 WEB_RELEVANCE_FLOOR）")

    # 当前 crag.py 默认阈值在 eval 集上的预测分布（即 Phase D 的 LLM 裁决量）
    from api.crag import CORRECT_THRESHOLD as _ct, INCORRECT_THRESHOLD as _it

    _pc = defaultdict(int)
    for item in cached:
        for d, _ in item["labels"]:
            _pc[_label_proxy(d.score, _ct, _it).value] += 1
    _tot = sum(_pc.values()) or 1
    print(f"\n[crag-eval] 当前 crag.py 默认阈值 (ct={_ct}, it={_it}) 在 eval 集预测分布:")
    print(f"           correct={_pc.get('correct',0)} / incorrect={_pc.get('incorrect',0)} "
          f"/ ambiguous={_pc.get('ambiguous',0)} "
          f"(模糊带 {_pc.get('ambiguous',0)/_tot*100:.1f}% — 即 Phase D LLM 仅裁决此比例文档)")

    if args.dump:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RESULTS_DIR / f"crag_calib_{ts}.json"
        dump = {
            "suggested": {
                "correct_t": best["correct_t"], "incorrect_t": best["incorrect_t"],
                "web_relevance_floor": 0.5,
            },
            "score_dist": {
                "correct": correct_scores,
                "incorrect": incorrect_scores,
            },
            "queries": [],
            "web_probes": [
                {"query": pq, "fired": f, "n_incorrect_top5": ni, "max_score_top5": round(ms, 4)}
                for pq, f, ni, ms in probe_rows
            ],
        }
        for item in cached:
            dump["queries"].append({
                "query": item["q"]["query"],
                "need_web": item["need_web"],
                "docs": [
                    {"slug": d.slug, "score": round(d.score or 0.0, 4), "is_correct": ic}
                    for d, ic in item["labels"]
                ],
            })
        out.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[crag-eval] 原始分数已写入 {out}")


if __name__ == "__main__":
    main()
