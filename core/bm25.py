"""In-memory BM25 index over the current Qdrant collection."""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi

from core.config import get_settings
from core.qdrant_client import get_qdrant
from data.parse_hexo import DocumentChunk


def tokenize(text: str) -> List[str]:
    """Tokenize mixed Chinese and English retrieval text."""
    value = (text or "").lower()
    tokens = [token.strip() for token in jieba.lcut(value) if token.strip()]
    tokens.extend(re.findall(r"[a-z0-9][a-z0-9._+-]*", value))
    return tokens


def _load_chunks() -> List[DocumentChunk]:
    client = get_qdrant()
    settings = get_settings()
    chunks: List[DocumentChunk] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            timeout=settings.QDRANT_READ_TIMEOUT,
        )
        for point in points:
            raw = (point.payload or {}).get("chunk")
            if raw:
                chunks.append(DocumentChunk.from_payload(raw))
        if offset is None:
            break
    return chunks

# BM25索引是运行中的后端进程，在内存中用qdrant的数据复印出来的一份全文检索结构
@lru_cache(maxsize=1)
def get_bm25_index() -> Tuple[BM25Okapi, Tuple[DocumentChunk, ...]]:
    """Build a process-local index; restart after recreating the collection."""
    # 全量scroll整个集合 不按照查询过滤，把整个集合导出一遍，
    # 在内存里建BM25索引，靠lru_cache缓存
    chunks = _load_chunks()
    # jieba全量分词
    corpus = [tokenize(chunk.embed_text()) for chunk in chunks]
    return BM25Okapi(corpus), tuple(chunks)


def search(query: str, limit: int) -> List[Tuple[DocumentChunk, float]]:
    """Return positive-scoring chunk matches, ordered by BM25 score."""
    index, chunks = get_bm25_index()
    scores = index.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [(chunks[position], float(score)) for position, score in ranked[:limit] if score > 0]

# 耗时2-5s
def warm_bm25() -> None:
    """Pre-build the BM25 index so the first request does not pay the cost."""
    logger = logging.getLogger("blog-rag")
    try:
        get_bm25_index()
        logger.info("bm25 index warmed")
    except Exception:
        logger.warning("bm25 warmup failed; first request will build on demand", exc_info=True)