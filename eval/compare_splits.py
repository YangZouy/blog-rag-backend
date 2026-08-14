# -*- coding: utf-8 -*-
"""Compare Agentic dev vs final evaluation results, surfacing real gaps vs noise.

The two frozen splits are now proportion-balanced, but final n is still small, so a
raw dev/final difference can be pure sampling noise. This script prints each metric
with both splits' bootstrap 95% CIs and flags whether the intervals overlap:

  * REAL   -> CIs do NOT overlap  -> the gap is unlikely to be noise; investigate.
  * noise  -> CIs overlap         -> the gap is within sampling error; don't over-fit.

It also prints a per-`type` retrieval table so you can see whether a gap lives in a
specific hard slice (e.g. multi_turn) rather than everywhere.

Usage
-----
  python eval/compare_splits.py <dev_result.json> <final_result.json>
  python eval/compare_splits.py --auto        # pick latest dev/final under eval/results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"

RETRIEVAL_METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "MRR", "slug_hit_rate"]
AGENTIC_METRICS = [
    "multi_turn_rewrite_accuracy",
    "sub_query_coverage",
    "expected_action_accuracy",
    "no_answer_refusal_accuracy",
    "citation_support_rate",
    "refusal_behavior_accuracy",
]


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(v, ci):
    if v is None:
        return "  -  "
    s = f"{v:.3f}"
    if ci:
        s += f" [{ci[0]:.2f},{ci[1]:.2f}]"
    return s


def _verdict(ci_dev, ci_final):
    if not ci_dev or not ci_final:
        return "n/a"
    lo_dev, hi_dev = ci_dev
    lo_fin, hi_fin = ci_final
    # overlap test on 95% intervals
    if hi_dev < lo_fin or hi_fin < lo_dev:
        return "REAL"
    return "noise"


def _auto_pick() -> tuple[str, str]:
    dev_files, final_files = [], []
    for f in glob.glob(str(RESULTS / "agentic_retrieval_results_*.json")):
        try:
            d = load(f)
        except Exception:
            continue
        split = (d.get("config", {}).get("split") or {})
        name = split.get("split") if isinstance(split, dict) else None
        ts = d.get("timestamp", "")
        if name == "dev":
            dev_files.append((ts, f))
        elif name == "final":
            final_files.append((ts, f))
    if not dev_files or not final_files:
        raise SystemExit("Could not auto-discover both dev and final result files under eval/results.")
    return max(dev_files)[1], max(final_files)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Agentic dev vs final evaluation results.")
    parser.add_argument("dev", nargs="?", help="dev result JSON")
    parser.add_argument("final", nargs="?", help="final result JSON")
    parser.add_argument("--auto", action="store_true", help="auto-pick latest dev/final under eval/results")
    args = parser.parse_args()
    if args.auto:
        dev_path, final_path = _auto_pick()
    elif args.dev and args.final:
        dev_path, final_path = args.dev, args.final
    else:
        parser.error("pass two result files or use --auto")

    dev = load(dev_path)
    final = load(final_path)
    dev_pr = dev["summary"]["post_orchestration_retrieval"]
    final_pr = final["summary"]["post_orchestration_retrieval"]
    dev_ag = dev["summary"]["agentic"]
    final_ag = final["summary"]["agentic"]

    print(f"DEV   : {os.path.basename(dev_path)}  (recall evaluated_rows={dev_pr.get('evaluated_rows')})")
    print(f"FINAL : {os.path.basename(final_path)}  (recall evaluated_rows={final_pr.get('evaluated_rows')})")

    print("\n=== Retrieval (post-orchestration) ===")
    print(f"{'metric':22} {'dev':>20} {'final':>20} {'delta':>8} {'verdict':>8}")
    for m in RETRIEVAL_METRICS:
        dv, dc = dev_pr.get(m), dev_pr.get("ci95", {}).get(m)
        fv, fc = final_pr.get(m), final_pr.get("ci95", {}).get(m)
        delta = (fv - dv) if isinstance(dv, (int, float)) and isinstance(fv, (int, float)) else None
        print(f"{m:22} {_fmt(dv, dc):>20} {_fmt(fv, fc):>20} {('' if delta is None else f'{delta:+.3f}'):>8} {_verdict(dc, fc):>8}")

    print("\n=== Agentic orchestration ===")
    print(f"{'metric':30} {'dev':>20} {'final':>20} {'delta':>8} {'verdict':>8}")
    for m in AGENTIC_METRICS:
        dv, dc = dev_ag.get(m), dev_ag.get("ci95", {}).get(m)
        fv, fc = final_ag.get(m), final_ag.get("ci95", {}).get(m)
        delta = (fv - dv) if isinstance(dv, (int, float)) and isinstance(fv, (int, float)) else None
        print(f"{m:30} {_fmt(dv, dc):>20} {_fmt(fv, fc):>20} {('' if delta is None else f'{delta:+.3f}'):>8} {_verdict(dc, fc):>8}")

    print("\n=== Retrieval by type (recall@1 / MRR, n) ===")
    types = sorted(set(dev_pr.get("by_type", {})) | set(final_pr.get("by_type", {})))
    print(f"{'type':22} {'dev@1':>8} {'devMRR':>8} {'dev n':>6} | {'fin@1':>8} {'finMRR':>8} {'fin n':>6}")
    for t in types:
        dm = dev_pr.get("by_type", {}).get(t)
        fm = final_pr.get("by_type", {}).get(t)
        d1 = f"{dm['recall@1']:.2f}" if dm else "  - "
        dM = f"{dm['MRR']:.2f}" if dm else "  - "
        dn = dm["n"] if dm else 0
        f1 = f"{fm['recall@1']:.2f}" if fm else "  - "
        fM = f"{fm['MRR']:.2f}" if fm else "  - "
        fn = fm["n"] if fm else 0
        print(f"{t:22} {d1:>8} {dM:>8} {dn:>6} | {f1:>8} {fM:>8} {fn:>6}")


if __name__ == "__main__":
    main()
