from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import List, Optional, Tuple
from functools import lru_cache
from api.models import SearchRequest  # noqa: F401  (kept for symmetry)
from core.config import get_settings
from core.embeddings import get_embeddings
from core.observability import timed_stage
from core.vector_store import get_vector_store
from data.parse_hexo import DocumentChunk

logger = logging.getLogger("blog-rag")


def _retry_network(fn, attempts: int = 3, base_delay: float = 0.5, label: str = "network"):
    """瞬时网络抖动（如经代理的 TLS 连接被对端重置 -> SSL: UNEXPECTED_EOF_WHILE_READING）
    重试封装。与下方 embedding 调用保持同一策略：递增退避，最多 attempts 次。

    fn 必须是无参可调用，且幂等（同一 query 重复调用结果一致）。
    发生重试时打 warning，便于从日志判断「是否发生重试」。
    """
    last_error = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 网络层异常类型繁杂，统一重试
            last_error = exc
            if i == attempts - 1:
                raise
            logger.warning("%s 重试 %d/%d: %s", label, i + 1, attempts, exc)
            time.sleep(base_delay * (i + 1))
    # 理论上不会走到这里（最后一次已 raise），仅作静态保险
    raise RuntimeError(f"{label} 失败（{attempts} 次）") from last_error


@lru_cache(maxsize=256)
def _embed_query_cached(query: str) -> tuple:
    """Query 向量缓存：同一问题（含追问拼接后的检索 query）只调一次智谱 API。

    返回 tuple（不可变）避免调用方意外修改缓存内容；用时转回 list。
    """
    vec = _retry_network(
        lambda: get_embeddings().embed_query(query),
        attempts=4,
        label="embed",
    )
    return tuple(vec)


def _embed_query(query: str) -> Tuple[list, bool]:
    """返回 (向量, 是否命中 lru_cache)。命中缓存 = 0 网络开销。"""
    hits_before = _embed_query_cached.cache_info().hits
    with timed_stage("embed_query", query=query) as f:
        vec = list(_embed_query_cached(query))
    f["cache_hit"] = _embed_query_cached.cache_info().hits > hits_before
    return vec, bool(f["cache_hit"])


def _faiss_search(qvec, limit: int, doc_type=None, tags=None) -> List[DocumentChunk]:
    with timed_stage("faiss_search", candidate_k=limit):
        return [c for c, _ in get_vector_store().search(qvec, limit, doc_type=doc_type, tags=tags)]


def _vector_path(query: str, limit: int, doc_type=None, tags=None) -> List[DocumentChunk]:
    """embedding + 向量检索串行（embedding 是远程调用，必须等其返回）。"""
    qvec, _ = _embed_query(query)
    return _faiss_search(qvec, limit, doc_type=doc_type, tags=tags)


def _bm25_search_timed(query: str, limit: int) -> List:
    from core.bm25 import search as bm25_search

    with timed_stage("bm25_search", candidate_k=limit):
        return bm25_search(query, limit)


# 纯向量检索（本地 Faiss，零网络）
def retrieve(
    query: str,
    top_k: int = 5,
    doc_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    candidate_k: Optional[int] = None,
) -> List[DocumentChunk]:
    s = get_settings()
    if candidate_k is None:
        candidate_k = s.RETRIEVAL_CANDIDATE_K
    limit = max(top_k, candidate_k)
    qvec, _ = _embed_query(query)
    return _faiss_search(qvec, limit, doc_type=doc_type, tags=tags)


# 向量+BM25融合
# 使用RRF（K = 60）把两套排名融合成一个分数重排
# 解决term类遗漏问题（TypeScript/BOM向量召回失败但BM25能命中的情况）
def retrieve_hybrid(
    query: str,
    top_k: int = 5,
    doc_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    candidate_k: Optional[int] = None,
) -> List[DocumentChunk]:
    """Fuse production vector retrieval and BM25 with reciprocal-rank fusion."""
    from core.bm25 import search as bm25_search

    settings = get_settings()
    limit = max(top_k, candidate_k or settings.RETRIEVAL_CANDIDATE_K)
    # 向量检索（含 embedding 远程调用）与 BM25 全文检索相互独立，
    # 用线程池并行执行，省掉串行等待，端到端延迟约降 200-400ms。
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_vec = ex.submit(_vector_path, query, limit, doc_type, tags)
        f_bm25 = ex.submit(_bm25_search_timed, query, limit)
        vector_chunks = f_vec.result()
        bm25_results = f_bm25.result()
    with timed_stage(
        "hybrid_fusion", vector_n=len(vector_chunks), bm25_n=len(bm25_results)
    ) as f:
        fused = {}
        for rank, chunk in enumerate(vector_chunks, start=1):
            fused[(chunk.slug, chunk.chunk_index)] = [chunk, 1.0 / (60 + rank)]
        for rank, (chunk, _score) in enumerate(bm25_results, start=1):
            if doc_type and chunk.doc_type != doc_type:
                continue
            if tags and not set(tags).intersection(chunk.tags or []):
                continue
            key = (chunk.slug, chunk.chunk_index)
            if key not in fused:
                fused[key] = [chunk, 0.0]
            fused[key][1] += 1.0 / (60 + rank)
        chunks = []
        for chunk, score in fused.values():
            chunk.score = score
            chunks.append(chunk)
        chunks = sorted(chunks, key=lambda chunk: chunk.score or 0.0, reverse=True)[:limit]
        f["fused_n"] = len(chunks)
    return chunks


# Hybrid候选+本地cross-encoder重排
def retrieve_with_rerank(
    query: str,
    top_k: int = 5,
    doc_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    candidate_k: Optional[int] = None,
) -> List[DocumentChunk]:
    """Retrieve hybrid candidates then rerank them with a cross-encoder."""
    from core.rerank import rerank

    settings = get_settings()
    limit = max(top_k, candidate_k or settings.RETRIEVAL_CANDIDATE_K)
    candidates = retrieve_hybrid(
        query,
        top_k=limit,
        doc_type=doc_type,
        tags=tags,
        candidate_k=max(limit, settings.RERANK_CANDIDATE_K),
    )
    with timed_stage("rerank", query=query, candidate_k=len(candidates), return_k=limit):
        return rerank(query, candidates, limit)
