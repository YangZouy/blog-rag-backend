"""Jina Reranker v2 multilingual API client."""
from __future__ import annotations

import logging
from typing import List

import httpx

from core.config import get_settings
from data.parse_hexo import DocumentChunk

logger = logging.getLogger("blog-rag")

JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
RERANK_API_TIMEOUT = 15

def _rerank_jina(query: str, chunks: List[DocumentChunk], limit: int) -> List[DocumentChunk]:
    s = get_settings()
    if not s.JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not set; cannot call the Jina reranker")

    resp = httpx.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {s.JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_RERANK_MODEL,
            "query": query,
            "documents": [c.embed_text() for c in chunks],
            "top_n": limit,
        },
        timeout=RERANK_API_TIMEOUT,
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
    try:
        return _rerank_jina(query, chunks, limit)
    except Exception:
        logger.exception("Jina reranking failed; falling back to hybrid score order")
        return sorted(chunks, key=lambda c: c.score or 0.0, reverse=True)[:limit]


def warm_reranker() -> None:
    """Keep the application startup hook stable; the remote API needs no warmup."""
    logger.info("reranker backend=jina (API); no local warmup needed")
