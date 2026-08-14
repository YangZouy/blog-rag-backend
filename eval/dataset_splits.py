# -*- coding: utf-8 -*-
"""Validate and resolve frozen development/final evaluation splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AGENTIC_DATASET = ROOT / "eval" / "agentic_eval_queries.json"
SPLIT_MANIFEST = ROOT / "eval" / "agentic_eval_splits.json"
SPLITS = ("dev", "final")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_agentic_split(split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a validated subset without duplicating the canonical questions."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose one of: {', '.join(SPLITS)}")

    dataset = _load_json(AGENTIC_DATASET)
    manifest = _load_json(SPLIT_MANIFEST)
    queries = dataset.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("Agentic dataset must contain a non-empty 'queries' list")
    by_id = {row.get("id"): row for row in queries if isinstance(row, dict)}
    if len(by_id) != len(queries) or None in by_id:
        raise ValueError("Agentic dataset query ids must be present and unique")

    actual_sha = sha256(AGENTIC_DATASET)
    expected_sha = manifest.get("source_dataset", {}).get("sha256")
    if actual_sha != expected_sha:
        raise ValueError(
            "Agentic source dataset changed after the split was frozen. "
            "Update agentic_eval_splits.json deliberately with a new stratified split; "
            "do not silently reuse the final validation set."
        )

    split_ids = manifest.get("splits", {})
    dev_ids = split_ids.get("dev")
    final_ids = split_ids.get("final")
    if not all(isinstance(ids, list) and ids for ids in (dev_ids, final_ids)):
        raise ValueError("Both dev and final split ids must be non-empty lists")
    if len(set(dev_ids)) != len(dev_ids) or len(set(final_ids)) != len(final_ids):
        raise ValueError("Split ids must not contain duplicates")
    if set(dev_ids) & set(final_ids):
        raise ValueError("Development and final split ids must be disjoint")
    if set(dev_ids) | set(final_ids) != set(by_id):
        raise ValueError("Split ids must cover every canonical Agentic query exactly once")

    selected_ids = split_ids[split]
    missing = [query_id for query_id in selected_ids if query_id not in by_id]
    if missing:
        raise ValueError(f"Split {split!r} refers to unknown query ids: {missing}")

    metadata = {
        "split": split,
        "record_count": len(selected_ids),
        "source_path": str(AGENTIC_DATASET.relative_to(ROOT)),
        "source_sha256": actual_sha,
        "manifest_path": str(SPLIT_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha256(SPLIT_MANIFEST),
        "frozen": True,
    }
    return [by_id[query_id] for query_id in selected_ids], metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the frozen Agentic evaluation split.")
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    args = parser.parse_args()
    names = SPLITS if args.split == "all" else (args.split,)
    for name in names:
        rows, metadata = load_agentic_split(name)
        print(json.dumps({"ids": [row["id"] for row in rows], **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
