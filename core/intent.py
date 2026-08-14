from __future__ import annotations
import re
from enum import Enum
from functools import lru_cache

from core.llm import get_classify_llm
from core.config import get_settings


class Intent(str, Enum):
    RAG = "rag"      # 需要检索本站资料后回答
    CHAT = "chat"    # 问候/感谢/闲聊/身份，不检索
    LIVE = "live"    # 实时数据（天气/股价/新闻），无工具，直接拒答


# ---------- 第一层：规则快路（零成本、即时、不会误判）----------
_GREETING = re.compile(r"^\s*(你好|您好|hi|hello|嗨|哈喽|在吗|在不在|有人吗)\b", re.I)
_THANKS = re.compile(r"(谢谢|感谢|多谢|thanks|thank\s*you)", re.I)
_IDENTITY = re.compile(r"(你是谁|你叫什么|你的名字|你是什么(东西|助手|机器人|ai|程序)|介绍一下你|你多大了|你是真人吗|你是男是女|你几岁)", re.I)
_CHITCHAT = re.compile(r"(陪我聊|讲个笑话|你喜欢|你觉得.*吗|今天心情|你吃饭了吗|晚安|早安)", re.I)

_LIVE_PATTERNS = (
    r"(?:今天|现在|当前|实时|最新|明天|后天).{0,10}(?:天气|气温|温度|降雨|下雨|空气质量|股价|股票|汇率|金价|油价|新闻|路况|比赛|彩票)",
    r"(?:天气|气温|温度|降雨|下雨|空气质量|股价|股票|汇率|金价|油价).{0,12}(?:怎么样|如何|多少|查询|预报|吗|走势|行情)",
    r"(?:股价|股票|汇率|金价|油价|彩票).{0,8}(?:是多少|涨了|跌了|今天)",
    r"(?:现在|当前|实时|最新).{0,16}(?:服务状态|运行状态|是否正常|可用性)",
)
_LIVE_RE = [re.compile(p, re.I) for p in _LIVE_PATTERNS]


def rule_intent(query: str) -> "Intent | None":
    """规则快路：命中即判定，未命中返回 None 交给 LLM。"""
    q = (query or "").strip()
    if not q:
        return Intent.CHAT
    if any(p.search(q) for p in _LIVE_RE):
        return Intent.LIVE
    if _GREETING.match(q) or _THANKS.search(q) or _IDENTITY.search(q) or _CHITCHAT.search(q):
        return Intent.CHAT
    return None


# ---------- 第二层：LLM 分类（glm-4-flash，便宜快）----------
_SYSTEM = (
    "你是意图分类器。判断用户问题属于哪一类，只输出一个英文标签，不要解释、不要标点。\n"
    "rag  —— 问题可能需要基于『博客文章内容』来回答（技术、项目、博主写过或可能写过的任何话题，"
    "包括 AI/RAG/前端/后端/算法/健身/随笔等；即使你也能从通用知识回答，只要像个人博客会覆盖的话题就选 rag）\n"
    "chat —— 问候、感谢、闲聊、询问 AI 助手身份等社交性内容，不需要检索博客\n"
    "live —— 需要实时外部数据（天气/股价/新闻/路况/汇率等）的问题\n"
    "输出只能是：rag 或 chat 或 live"
)


@lru_cache(maxsize=512)
def _llm_classify(query: str) -> Intent:
    try:
        resp = get_classify_llm().invoke(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": query}]
        )
        text = (resp.content or "").strip().lower()
        if "live" in text:
            return Intent.LIVE
        if "chat" in text:
            return Intent.CHAT
        return Intent.RAG
    except Exception:
        return Intent.RAG  # 分类失败保守走 rag，保证能回答


def classify_intent(query: str) -> Intent:
    """先规则快路，规则判不了再 LLM。"""
    ruled = rule_intent(query)
    if ruled is not None:
        return ruled
    return _llm_classify(query)
