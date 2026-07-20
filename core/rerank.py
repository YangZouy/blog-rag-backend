"""Lazy cross-encoder reranking for a small set of retrieved chunks."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

from data.parse_hexo import DocumentChunk

DEFAULT_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(os.getenv("RERANK_MODEL", DEFAULT_MODEL))


def rerank(query: str, chunks: List[DocumentChunk], limit: int) -> List[DocumentChunk]:
    """Score hybrid candidates with a cross-encoder and retain the best ones."""
    if not chunks:
        return []
    scores = _model().predict([(query, chunk.embed_text()) for chunk in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda item: float(item[1]), reverse=True)
    out = []
    for chunk, score in ranked[:limit]:
        chunk.score = float(score)
        out.append(chunk)
    return out


def warm_reranker() -> None:
    """Pre-load the cross-encoder model so the first request does not pay the cost."""
    logger = logging.getLogger("blog-rag")
    try:
        _model()
        logger.info("reranker model warmed")
    except Exception:
        logger.warning("reranker warmup failed; first request will load on demand", exc_info=True)