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

@lru_cache
def get_classify_llm() -> ChatOpenAI:
    """意图分类用 glm-4-flash：快、便宜、temperature=0 稳定。"""
    s = get_settings()  # ← 修：原来漏了 ()
    return ChatOpenAI(
        model=s.INTENT_LLM_MODEL,
        api_key=s.ZHIPU_API_KEY,
        base_url=s.ZHIPU_BASE_URL,  # ← 修：现在 config 里有了
        temperature=0,
        max_tokens=8,
    )


@lru_cache
def get_planning_llm() -> ChatOpenAI:
    """Short structured planning calls for complex-question decomposition."""
    s = get_settings()
    return ChatOpenAI(
        model=s.INTENT_LLM_MODEL,
        api_key=s.ZHIPU_API_KEY,
        base_url=s.ZHIPU_BASE_URL,
        temperature=0,
        max_tokens=180,
    )

@lru_cache
def get_summarize_llm() -> ChatOpenAI:
    """自动摘要用 glm-4-flash：一句话概括，temperature=0 保持简洁。"""
    s = get_settings()
    return ChatOpenAI(
        model=s.INTENT_LLM_MODEL,
        api_key=s.ZHIPU_API_KEY,
        base_url=s.ZHIPU_BASE_URL,
        temperature=0,
        max_tokens=40,
    )
