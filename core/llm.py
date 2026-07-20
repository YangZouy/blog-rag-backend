"""ChatOpenAI 单例客户端。

两层架构（成本优化）：

grade_llm -> 廉价模型，用于相关性评分 / 查询改写

gen_llm -> 更强模型，用于最终答案合成
两者均兼容 OpenAI 接口，因此任何厂商（DeepSeek / 智谱 / 通义 / 硅基流动）
均可通过配置接入。
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from core.config import get_settings

"""
python标准库里的装饰器，用来做缓存
表示：第一次调用get_grade_llm时，真正创建一个LLM客户端对象
以后再调用，直接复用之前那个，不重复创建。
"""
@lru_cache
def get_grade_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.GRADE_MODEL,
        api_key=s.grade_api_key,
        base_url=s.GRADE_BASE_URL,
        temperature=0,
    )


@lru_cache
def get_gen_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.GEN_MODEL,
        api_key=s.DEEPSEEK_API_KEY,
        base_url=s.GEN_BASE_URL,
        temperature=0,
    )
