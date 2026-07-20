"""最终答案生成模型的 ChatOpenAI 单例客户端。"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from core.config import get_settings

@lru_cache
def get_gen_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.GEN_MODEL,
        api_key=s.DEEPSEEK_API_KEY,
        base_url=s.GEN_BASE_URL,
        temperature=0,
    )
