"""Qdrant 客户端单例 + 集合初始化引导。

集合的向量维度必须等于 EMBED_DIM（智谱 embedding-3 默认为 2048）。
更换嵌入模型时需同步更新 EMBED_DIM，否则写入/搜索操作会因维度不匹配而失败。
"""
from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from core.config import get_settings

logger = logging.getLogger("blog-rag")


@lru_cache
def get_qdrant() -> QdrantClient:
    s = get_settings()
    if s.QDRANT_URL:
        return QdrantClient(
            url=s.QDRANT_URL,
            api_key=s.QDRANT_API_KEY or None,
            timeout=s.QDRANT_READ_TIMEOUT,
        )
    # 直接在python进程中开一个临时的、内存版的Qdrant 数据在内存中
    return QdrantClient(path=":memory:")


def warm_qdrant() -> None:
    """Warm the same Qdrant query path used by retrieval before serving requests."""
    s = get_settings()
    if not s.QDRANT_URL or not s.QDRANT_WARMUP_ENABLED:
        return
    try:
        get_qdrant().query_points(
            collection_name=s.QDRANT_COLLECTION,
            query=[0.0] * s.EMBED_DIM,
            limit=1,
            with_payload=False,
            timeout=s.QDRANT_READ_TIMEOUT,
        )
    except Exception:
        logger.warning("qdrant warmup failed; first request will retry normally", exc_info=True)

"""
指定重建原因：
向量维度改变 
距离度量变了 
清空旧数据，重新全量导入
"""
def ensure_collection(client: QdrantClient | None = None, recreate: bool = False) -> None:
    """Create the collection if it does not exist (or recreate when asked)."""
    s = get_settings()
    client = client or get_qdrant()
    if recreate:
        client.recreate_collection(
            collection_name=s.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=s.EMBED_DIM, distance=Distance.COSINE),
        )
        return
    try:
        client.get_collection(s.QDRANT_COLLECTION)
    except Exception:
        client.create_collection(
            collection_name=s.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=s.EMBED_DIM, distance=Distance.COSINE),
        )
