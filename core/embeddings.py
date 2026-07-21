"""Singleton embedding client.

后端可切换：
- zhipu（默认）：远程 OpenAI 兼容 API（智谱 embedding-3，2048 维）
- local：本地 sentence-transformers 模型（如 BAAI/bge-small-zh-v1.5，512 维）

两套后端对外暴露同一接口：embed_query(text) -> List[float]、
embed_documents(texts, chunk_size=...) -> List[List[float]]，
因此 retriever / ingest 无需关心具体后端。
切换只需改 EMBED_BACKEND（及对应的 EMBED_DIM / QDRANT_COLLECTION）。
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from core.config import get_settings


class _ZhipuEmbeddings:
    """远程 OpenAI 兼容嵌入（默认）。"""

    def __init__(self, settings) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._inner = OpenAIEmbeddings(
            model=settings.EMBED_MODEL,
            api_key=settings.ZHIPU_API_KEY,
            base_url=settings.EMBED_BASE_URL,
            # 纵深防御：经代理访问时偶发 TLS 连接重置，让 openai SDK 也重试一次连接层错误
            max_retries=3,
        )

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(text)

    def embed_documents(self, texts: List[str], chunk_size: int = 64) -> List[List[float]]:
        return self._inner.embed_documents(texts, chunk_size=chunk_size)


class _LocalEmbeddings:
    """本地 sentence-transformers 嵌入（小维度、零网络、低延迟）。"""

    def __init__(self, settings) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(settings.EMBED_LOCAL_MODEL)
        # bge 系列建议给查询加前缀，文档侧不加（提升检索对齐质量）
        self._prefix = settings.EMBED_LOCAL_QUERY_PREFIX

    def embed_query(self, text: str) -> List[float]:
        vec = self._model.encode(
            self._prefix + text, normalize_embeddings=True
        )
        return vec.tolist()

    def embed_documents(self, texts: List[str], chunk_size: int = 32) -> List[List[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=chunk_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


@lru_cache
def get_embeddings():
    """返回当前配置下的嵌入客户端单例。"""
    s = get_settings()
    if (s.EMBED_BACKEND or "zhipu").lower() == "local":
        return _LocalEmbeddings(s)
    return _ZhipuEmbeddings(s)
