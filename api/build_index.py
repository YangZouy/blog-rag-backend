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
    articles = list(by_slug.values())
    # 校验：URL 里若残留非 ASCII 字节、以 // 结尾、或 post 缺少日期段，多半是标题没匹配上
    # search.xml 且拼音兜底不准 → 线上 404。建索引时显式告警，避免坏链静默进向量库。
    suspicious = [
        a for a in articles
        if any(ord(c) > 127 for c in a["url"])
        or a["url"].endswith("//")
        or (a["doc_type"] == "post" and a["url"].count("/") < 6)
    ]
    if suspicious:
        print(f"⚠️  {len(suspicious)} 篇文章 URL 可疑（可能 404），请检查其标题是否已在 search.xml 中：")
        for a in suspicious:
            print(f"    - {a['title']!r} -> {a['url']}")

    index = {
        "generated_from": repo_path,
        "count": len(by_slug),
        "articles": articles,
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
