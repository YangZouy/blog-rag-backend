from __future__ import annotations

import json
import os
import logging
import re
from data.parse_hexo import parse_hexo_repo
from core.llm import get_summarize_llm

logger = logging.getLogger("ingest")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "data", "blog_index.json")

_SUM_PROMPT = (
    "用一句中文（不超过30字）概括下面这篇博客文章的主题，"
    "只输出这句话本身，不要标点结尾、不要解释：\n"
    "标题：{title}\n内容片段：{excerpt}"
)

def _summarize_article(article: dict) -> str | None:
    try:
        llm = get_summarize_llm()
        resp = llm.invoke(
            [{"role": "user", "content": _SUM_PROMPT.format(
                title=article["title"], excerpt=article.get("excerpt", ""))}]
        )
        text = (resp.content or "").strip()
        text = re.sub(r"[。.！!?？\s]+$", "", text)  # 去结尾标点/空白
        return text[:40]
    except Exception:
        logger.warning("summarize failed for %s", article.get("slug"), exc_info=True)
        return None

def _load_prev_summaries(out_path: str) -> dict:
    """读旧 blog_index.json 里的 summary 映射，未变文章复用，省 LLM 调用。"""
    import os
    if os.path.isfile(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
            return {a["slug"]: a.get("summary") for a in data.get("articles", [])}
        except Exception:
            return {}
    return {}

def build_index(
    repo_path: str,
    out_path: str = DEFAULT_OUT,
    summarize: bool = False,
    changed_slugs: set[str] | None = None,
) -> dict:
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
            "description": c.description
        }
    articles = list(by_slug.values())

    if summarize:
        for a in articles:
            a["summary"] = a.get("description") or _summarize_article(a)

    # 原有 suspicious URL 告警逻辑保持原样（省略，原样保留）...
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

# 命令行入口
if __name__ == "__main__":
    main()
