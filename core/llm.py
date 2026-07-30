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

from core.config import get_settings
_classify_llm = None

def get_classify_llm():
    """意图分类用 glm-4-flash：快、便宜、temperature=0 稳定"""
    global _classify_llm
    if _classify_llm is None:
        s = get_settings
        _classify_llm = ChatOpenAI(
            model=s.INTENT_LLM_MODEL,
            api_key=s.ZHIPU_API_KEY,
            base_url=s.ZHIPU_BASE_URL,
            # 关掉随机性，限制输出长度
            # 判别类：同样的输入保持相同的输出
            temperature=0,
            max_tokens=8,  
        )
    return _classify_llm