"""Singleton embedding client.

后端：智谱 embedding-3 远程 OpenAI 兼容 API（2048 维）。retriever / ingest 统一通过
get_embeddings() 拿客户端，对外暴露 embed_query / embed_documents 两接口。
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from core.config import get_settings


class _ZhipuEmbeddings:
    """远程 OpenAI 兼容嵌入（智谱 embedding-3）。"""

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


@lru_cache
def get_embeddings():
    """返回智谱嵌入客户端单例。"""
    return _ZhipuEmbeddings(get_settings())
