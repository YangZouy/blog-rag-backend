"""Vector retrieval from Qdrant.

Each point stores the serialized DocumentChunk in its payload["chunk"], so we
can reconstruct typed chunks directly. Optional payload filtering by doc_type
and tags is supported.
"""
from __future__ import annotations

import time
from typing import List, Optional

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from api.models import SearchRequest  # noqa: F401  (kept for symmetry)
from core.config import get_settings
from core.embeddings import get_embeddings
from core.qdrant_client import get_qdrant
from data.parse_hexo import DocumentChunk

"""
查询向量数据库的过滤条件
FieldCondition表示对某个字段加条件
MatchValue(value=doc_type) 精确匹配单个值，类似“等于”
MatchAny(any=tags) 匹配多个候选值中的任意一个，类似 “in (...)”
must=conditions 表示这些条件都必须满足，类似 AND
"""
def _retry_network(fn, attempts: int = 3, base_delay: float = 0.5, label: str = "network"):
    """瞬时网络抖动（如经代理的 TLS 连接被对端重置 -> SSL: UNEXPECTED_EOF_WHILE_READING）
    重试封装。与下方 Qdrant query 的已有重试保持同一策略：递增退避，最多 attempts 次。

    fn 必须是无参可调用，且幂等（同一 query 重复调用结果一致）。
    """
    last_error = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 网络层异常类型繁杂，统一重试
            last_error = exc
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (i + 1))
    # 理论上不会走到这里（最后一次已 raise），仅作静态保险
    raise RuntimeError(f"{label} 失败（{attempts} 次）") from last_error


def _build_filter(doc_type: Optional[str], tags: Optional[List[str]]) -> Optional[Filter]:
    conditions = []
    if doc_type:
        conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
    if tags:
        conditions.append(FieldCondition(key="tags", match=MatchAny(any=tags)))
    if not conditions:
        return None
    return Filter(must=conditions)


def retrieve(
    query: str,
    top_k: int = 5,
    doc_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    candidate_k: Optional[int] = None,
) -> List[DocumentChunk]:
    s = get_settings()
    # 先拉大候选池（top_k 与候选池取大者），再交给 LLM 筛选，避免漏掉长尾相关片段
    if candidate_k is None:
        candidate_k = s.RETRIEVAL_CANDIDATE_K
    limit = max(top_k, candidate_k)
    # embedding 调用经代理/远端时可能出现瞬时 TLS 重置（SSL EOF），无内置重试会直接失败；
    # 用退避重试兜住，与下方 Qdrant 查询重试策略一致。
    qvec = _retry_network(
        lambda: get_embeddings().embed_query(query),
        attempts=4,
        label="embed",
    )
    client = get_qdrant()
    """
    query_filter表示在向量相似的搜索基础上，再加结构化过滤条件
    collection中存储的是向量（文章chunk的embedding）
    和payload（附加字段：文章标题、URL、内容、tags、doc_type等元数据）

    resp返回 匹配到的向量点，每个点通常会包含
    id score vector payload等
    """
    resp = None
    last_error = None
    for attempt in range(3):
        try:
            resp = client.query_points(
                collection_name=s.QDRANT_COLLECTION,
                query=qvec,
                limit=limit,
                query_filter=_build_filter(doc_type, tags),
                with_payload=True,
                timeout=s.QDRANT_READ_TIMEOUT,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    if resp is None:
        raise RuntimeError("Qdrant query returned no response") from last_error
    chunks: List[DocumentChunk] = []
    # resp.points：一组命中的向量点，包含id，score，payload等
    for p in resp.points:
        payload = p.payload or {}
        chunk_raw = payload.get("chunk")
        if chunk_raw:
            c = DocumentChunk.from_payload(chunk_raw)
            c.score = p.score  # 保留向量相似度分数，供后续阈值判断与排序
            chunks.append(c)
    return sorted(chunks, key=lambda chunk: chunk.score or 0.0, reverse=True)


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
    vector_chunks = retrieve(
        query,
        top_k=limit,
        doc_type=doc_type,
        tags=tags,
        candidate_k=limit,
    )
    fused = {}
    for rank, chunk in enumerate(vector_chunks, start=1):
        fused[(chunk.slug, chunk.chunk_index)] = [chunk, 1.0 / (60 + rank)]
    for rank, (chunk, _score) in enumerate(bm25_search(query, limit), start=1):
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
    return sorted(chunks, key=lambda chunk: chunk.score or 0.0, reverse=True)[:limit]

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
        candidate_k=max(limit, 50),
    )
    return rerank(query, candidates, limit)
