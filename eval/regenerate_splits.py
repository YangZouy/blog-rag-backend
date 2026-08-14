# -*- coding: utf-8 -*-
"""Regenerate a *proportion-balanced* development/final split for the Agentic dataset.

Why this exists
---------------
The previous frozen split only guaranteed *coverage* (every type appears in both
splits) but not *proportional balance*. That produced an inverted-difficulty
artifact: dev was ~75% single_hop (easy) while final was ~87% multi_turn/complex
(hard). Comparing aggregate metrics across such splits is apples-to-oranges.

This script stratifies by `type` so dev and final carry the **same proportion** of
each type, making aggregate dev/final comparisons meaningful. It still preserves:
  * type coverage in both splits (every type with count>=2 lands in both),
  * the rule that authorization_boundary cases stay in dev (safety regression).

Determinism
-----------
Assignment within a type is by sorted id, so re-running gives identical output.
The source dataset is NOT modified, so its sha256 is unchanged and the integrity
check in dataset_splits.py keeps passing.

Usage
-----
    python eval/regenerate_splits.py            # prints stats + writes manifest
    python eval/regenerate_splits.py --dry-run  # print only, do not write
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "agentic_eval_queries.json"
MANIFEST = ROOT / "eval" / "agentic_eval_splits.json"
DEV_RATIO = 0.7
FORCED_DEV_TYPES = {"authorization_boundary"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a proportion-balanced Agentic split.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing.")
    args = parser.parse_args()

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    queries = dataset.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("Agentic dataset must contain a non-empty 'queries' list")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        by_type[q.get("type", "unknown")].append(q)

    dev_ids: list[str] = []
    final_ids: list[str] = []
    for type_name, items in by_type.items():
        items_sorted = sorted(items, key=lambda q: q["id"])
        n = len(items_sorted)
        if type_name in FORCED_DEV_TYPES:
            d = n  # all to dev
        else:
            d = round(DEV_RATIO * n)
            if n - d < 1:  # keep at least one in final
                d = n - 1
        if d < 1:  # keep at least one in dev
            d = 1
        dev_ids += [q["id"] for q in items_sorted[:d]]
        final_ids += [q["id"] for q in items_sorted[d:]]

    # Fix the total to 56/24 by shuttling single_hop between splits. single_hop is
    # present in both by construction, so this never breaks type coverage.
    total_dev_target = round(DEV_RATIO * len(queries))
    sh_dev = [i for i in dev_ids if _type_of(by_type, i) == "single_hop"]
    sh_final = [i for i in final_ids if _type_of(by_type, i) == "single_hop"]
    while len(dev_ids) > total_dev_target and sh_final:
        move = sh_final.pop()
        final_ids.remove(move)
        dev_ids.append(move)
    while len(dev_ids) < total_dev_target and sh_dev:
        move = sh_dev.pop()
        dev_ids.remove(move)
        final_ids.append(move)

    dev_ids.sort()
    final_ids.sort()

    # Stats
    print(f"Total queries: {len(queries)}  ->  dev={len(dev_ids)} final={len(final_ids)}")
    print(f"{'type':24} {'total':>6} {'dev':>6} {'final':>6} {'dev%':>6} {'final%':>7}")
    for type_name in sorted(by_type):
        tot = len(by_type[type_name])
        dv = sum(1 for i in dev_ids if _type_of(by_type, i) == type_name)
        fn = sum(1 for i in final_ids if _type_of(by_type, i) == type_name)
        print(f"{type_name:24} {tot:>6} {dv:>6} {fn:>6} {100*dv/tot:>5.1f}% {100*fn/tot:>6.1f}%")

    if args.dry_run:
        print("\n[dry-run] manifest not written.")
        return

    old = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    backup = MANIFEST.with_suffix(".json.bak")
    if MANIFEST.exists():
        backup.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\nBacked up previous manifest -> {backup.name}")

    manifest = {
        "version": (old.get("version", 1) + 1) if isinstance(old.get("version"), int) else 3,
        "purpose": "Proportion-balanced development/final split for Agentic orchestration evaluation.",
        "source_dataset": {
            "path": "eval/agentic_eval_queries.json",
            "record_count": len(queries),
            "sha256": sha256(DATASET),
        },
        "strategy": {
            "development_ratio": DEV_RATIO,
            "development_count": len(dev_ids),
            "final_count": len(final_ids),
            "rule": (
                "Tune rewrite, planning, evidence, and remediation settings on dev only. "
                "Freeze configuration before inspecting final per-case outcomes. "
                "Splits are STRATIFIED BY PROPORTION (each type keeps ~70/30 dev/final), "
                "so aggregate dev/final comparisons are meaningful; always report per-type "
                "breakdowns + bootstrap CIs (eval/compare_splits.py) because final n is small."
            ),
            "stratification": (
                "Every type with count>=2 appears in BOTH splits at ~70/30 ratio. "
                "authorization-boundary cases remain in development as a safety regression check."
            ),
        },
        "splits": {"dev": dev_ids, "final": final_ids},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {MANIFEST.name} (version {manifest['version']}).")


def _type_of(by_type: dict[str, list[dict]], qid: str) -> str:
    for t, items in by_type.items():
        if any(q["id"] == qid for q in items):
            return t
    return "unknown"


if __name__ == "__main__":
    main()
