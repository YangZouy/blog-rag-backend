"""Singleton OpenAI-compatible embedding client.

Swapping the embedding vendor is a one-line config change
(EMBED_MODEL / EMBED_BASE_URL / EMBED_DIM in core.config).
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from core.config import get_settings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.EMBED_MODEL,
        api_key=s.ZHIPU_API_KEY,
        base_url=s.EMBED_BASE_URL,
        # 纵深防御：经代理访问时偶发 TLS 连接重置，让 openai SDK 也重试一次连接层错误
        max_retries=3,
    )
