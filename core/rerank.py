"""Cross-encoder reranking — supports two backends:

  local  (default) — BAAI/bge-reranker-base via sentence-transformers
  jina             — Jina Reranker v2 multilingual API (recommended for Vercel;
                     eliminates PyTorch cold-start / 250 MB size limit issues)

Set RERANK_BACKEND=jina and JINA_API_KEY in .env to switch to the API backend.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

import httpx

from core.config import get_settings
from data.parse_hexo import DocumentChunk

logger = logging.getLogger("blog-rag")

DEFAULT_LOCAL_MODEL = "BAAI/bge-reranker-base"


# ---------------------------------------------------------------------------
# Local backend (sentence-transformers CrossEncoder)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(os.getenv("RERANK_MODEL", DEFAULT_LOCAL_MODEL))


def _rerank_local(query: str, chunks: List[DocumentChunk], limit: int) -> List[DocumentChunk]:
    scores = _local_model().predict([(query, c.embed_text()) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)
    out = []
    for chunk, score in ranked[:limit]:
        chunk.score = float(score)
        out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Jina API backend
# ---------------------------------------------------------------------------

def _rerank_jina(query: str, chunks: List[DocumentChunk], limit: int) -> List[DocumentChunk]:
    s = get_settings()
    if not s.JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not set; cannot use RERANK_BACKEND=jina")

    resp = httpx.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {s.JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": s.JINA_RERANK_MODEL,
            "query": query,
            "documents": [c.embed_text() for c in chunks],
            "top_n": limit,
        },
        timeout=s.RERANK_API_TIMEOUT,
    )
    resp.raise_for_status()

    results = resp.json()["results"]
    out: List[DocumentChunk] = []
    for r in results:
        chunk = chunks[r["index"]]
        chunk.score = float(r["relevance_score"])
        out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def rerank(query: str, chunks: List[DocumentChunk], limit: int) -> List[DocumentChunk]:
    """Score hybrid candidates and return the top `limit` chunks."""
    if not chunks:
        return []
    backend = get_settings().RERANK_BACKEND.lower()
    try:
        if backend == "jina":
            return _rerank_jina(query, chunks, limit)
        return _rerank_local(query, chunks, limit)
    except Exception:
        logger.exception("rerank(%s) failed; falling back to hybrid score order", backend)
        return sorted(chunks, key=lambda c: c.score or 0.0, reverse=True)[:limit]


def warm_reranker() -> None:
    """Pre-load the reranker so the first request does not pay cold-start cost."""
    backend = get_settings().RERANK_BACKEND.lower()
    if backend == "jina":
        logger.info("reranker backend=jina (API); no warmup needed")
        return
    try:
        _local_model()
        logger.info("reranker model warmed (local)")
    except Exception:
        logger.warning("reranker warmup failed; first request will load on demand", exc_info=True)
