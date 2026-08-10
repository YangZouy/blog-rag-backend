"""In-memory BM25 index over the local Faiss vector store."""
from __future__ import annotations

import logging
import re
import time
from functools import lru_cache
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi

from core.config import get_settings
from core.vector_store import get_vector_store
from data.parse_hexo import DocumentChunk

logger = logging.getLogger("blog-rag")


def tokenize(text: str) -> List[str]:
    """Tokenize mixed Chinese and English retrieval text."""
    value = (text or "").lower()
    tokens = [token.strip() for token in jieba.lcut(value) if token.strip()]
    tokens.extend(re.findall(r"[a-z0-9][a-z0-9._+-]*", value))
    return tokens


def _load_chunks() -> List[DocumentChunk]:
    # 直接从本地 Faiss 存储取全量 chunk（等价于原先从向量库 scroll 全量）
    return get_vector_store().get_all()


# BM25索引是运行中的后端进程，在内存中用本地存储的数据复印出来的一份全文检索结构
@lru_cache(maxsize=1)
def get_bm25_index() -> Tuple[BM25Okapi, Tuple[DocumentChunk, ...]]:
    """Build a process-local index; restart after recreating the vector store."""
    # 全量加载整个集合 不按照查询过滤，把整个集合导出一遍，
    # 在内存里建BM25索引，靠lru_cache缓存
    t0 = time.perf_counter()
    chunks = _load_chunks()
    # jieba全量分词
    corpus = [tokenize(chunk.embed_text()) for chunk in chunks]
    index = BM25Okapi(corpus)
    build_ms = (time.perf_counter() - t0) * 1000
    logger.info("bm25 index built: %d chunks (build %.1fms)", len(chunks), build_ms)
    return index, tuple(chunks)


def search(query: str, limit: int) -> List[Tuple[DocumentChunk, float]]:
    """Return positive-scoring chunk matches, ordered by BM25 score."""
    index, chunks = get_bm25_index()
    scores = index.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [(chunks[position], float(score)) for position, score in ranked[:limit] if score > 0]


def warm_bm25() -> bool:
    """Pre-build the BM25 index so the first request does not pay the cost."""
    try:
        t0 = time.perf_counter()
        get_bm25_index()
        logger.info("bm25 index warmed (total %.1fms)", (time.perf_counter() - t0) * 1000)
        return True
    except Exception:
        logger.warning("bm25 warmup failed; first request will build on demand", exc_info=True)
        return False
