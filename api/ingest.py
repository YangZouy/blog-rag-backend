"""将 Hexo 源仓库导入本地 Faiss 向量存储。

本地运行：
    python -m api.ingest --repo /path/to/hexo-source

CI 中（参见 scripts/ingest_ci.sh）：先检出源仓库，然后调用此脚本。
写入操作是幂等的（chunk key = slug:chunk_index），因此重复运行是安全的。
"""
from __future__ import annotations

import argparse
import logging
import sys
import hashlib
import json
import os

from core.config import get_settings
from core.embeddings import get_embeddings
from core.vector_store import get_vector_store
from data.parse_hexo import parse_hexo_repo
from data.parse_pdf import parse_pdfs
from api.build_index import build_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")

def _group_by_slug(docs):
    """把同一slug的所有chunk归到一组"""
    grouped: dict[str, list] = {}
    for c in docs:
        grouped.setdefault(c.slug, []).append(c)
    return grouped

def _slug_hash(chunks) -> str:
    """给一篇（一个 slug 的所有 chunk）算一个稳定 hash。

    拼接 标题/标签/章节/正文 后 sha256。只要文章实质内容变了，hash 就变，
    从而判出"该 slug 需要重新嵌入"。
    """
    norm = []
    for c in sorted(chunks, key=lambda x: x.chunk_index):
        norm.append(f"{c.title}\n{c.url}\n{c.tags}\n{c.section}\n{c.content}")
    return hashlib.sha256("\n===\n".join(norm).encode("utf-8")).hexdigest()

def _load_state() -> dict:
    """读取上次入库留下的 slug→hash 状态。"""
    from core.config import get_settings
    p = get_settings().INGEST_STATE_PATH
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f).get("hashes", {})
        except Exception:
            return {}
    return {}

def _save_state(hashes: dict) -> None:
    from core.config import get_settings
    p = get_settings().INGEST_STATE_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"hashes": hashes}, f, ensure_ascii=False, indent=2)

def _delete_by_slugs(store, slugs) -> None:
    """按 slug 批量删除本地 Faiss 中的 chunk（用于"文章变了/删了"的清理）。"""
    store.delete_by_slug(slugs)

def run_ingest(
    repo_path: str,
    recreate: bool = False,
    incremental: bool = False,
    summarize: bool = False,
) -> int:
    s = get_settings()
    store = get_vector_store()
    if recreate:
        # --recreate：清空本地索引，从头全量重建
        store.reset()
    embed = get_embeddings()

    chunks = parse_hexo_repo(repo_path) + parse_pdfs(repo_path)
    docs = [c for c in chunks if c.content]
    if not docs:
        logger.warning("no ingestable documents found in %s", repo_path)
        return 0

    grouped = _group_by_slug(docs)
    current_hashes = {slug: _slug_hash(cs) for slug, cs in grouped.items()}

    changed: set[str] = set()
    removed: list[str] = []
    if incremental and not recreate:
        prev = _load_state()
        changed = {sl for sl, h in current_hashes.items() if prev.get(sl) != h}
        removed = [sl for sl in prev if sl not in current_hashes]
        logger.info(
            "incremental: %d changed, %d removed, %d unchanged",
            len(changed), len(removed), len(current_hashes) - len(changed),
        )
        target = {sl: cs for sl, cs in grouped.items() if sl in changed}
        # 关键：先删掉这些 slug 的全部旧 chunk，避免文章变短残留尾部 chunk
        _delete_by_slugs(store, list(changed) + removed)
    else:
        # 非增量的全量：仍清掉"磁盘已删"的 slug，防止陈旧残留
        target = grouped
        if not recreate:
            prev = _load_state()
            removed = [sl for sl in prev if sl not in current_hashes]
            _delete_by_slugs(store, removed)

    to_embed = [c for cs in target.values() for c in cs]
    if to_embed:
        logger.info(
            "embedding %d chunks (incremental=%s, batch=%d)...",
            len(to_embed), incremental, s.EMBED_BATCH_SIZE,
        )
        vectors = embed.embed_documents(
            [c.embed_text() for c in to_embed],
            chunk_size=s.EMBED_BATCH_SIZE,
        )
        store.upsert([(c, v) for c, v in zip(to_embed, vectors)])
        logger.info("upserted %d chunks", len(to_embed))
    else:
        logger.info("nothing to embed")

    _save_state(current_hashes)

    # 重建文章清单（含可选摘要）。增量模式只给"变化 slug"重新摘要，省 token。
    try:
        changed_slugs = (set(changed) | set(removed)) if incremental else None
        idx = build_index(
            repo_path, summarize=summarize, changed_slugs=changed_slugs
        )
        logger.info("built blog index: %d articles", idx["count"])
    except Exception:
        logger.warning("blog index build skipped (non-fatal)", exc_info=True)
    return len(to_embed)

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a Hexo repo into local Faiss")
    parser.add_argument("--repo", required=True, help="path to the Hexo repo root")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and recreate the local index before ingest",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="only embed changed/new slugs and delete removed ones (needs prior state)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="generate a one-line LLM summary per article into blog_index.json",
    )
    args = parser.parse_args()
    n = run_ingest(
        args.repo,
        recreate=args.recreate,
        incremental=args.incremental,
        summarize=args.summarize,
    )
    print(f"ingested {n} chunks")
    sys.exit(0 if n >= 0 else 1)

if __name__ == "__main__":
    main()
