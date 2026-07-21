"""Generate data/blog_index.json: one entry per article (de-duplicated from chunks).

This lightweight inventory (title / url / tags / doc_type / short excerpt) is
injected into the generation prompt so the LLM knows what the blog actually
contains — even when retrieval returns nothing relevant. It prevents the
"can't answer but still recommends / hallucinates a topic" failure mode by
giving the model a global map of the site.
"""
from __future__ import annotations

import json
import os

from data.parse_hexo import parse_hexo_repo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "data", "blog_index.json")


def build_index(repo_path: str, out_path: str = DEFAULT_OUT) -> dict:
    chunks = parse_hexo_repo(repo_path)
    by_slug: dict[str, dict] = {}
    for c in chunks:
        if c.slug in by_slug:
            continue
        by_slug[c.slug] = {
            "title": c.title,
            "url": c.url,
            "doc_type": c.doc_type,
            "tags": c.tags,
            "excerpt": (c.content or "").strip()[:100],
        }
    index = {
        "generated_from": repo_path,
        "count": len(by_slug),
        "articles": list(by_slug.values()),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    return index


def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Build blog article index (data/blog_index.json)")
    ap.add_argument("--repo", required=True, help="path to the Hexo repo root")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output json path")
    args = ap.parse_args()
    idx = build_index(args.repo, args.out)
    print(f"built index: {idx['count']} articles -> {args.out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
