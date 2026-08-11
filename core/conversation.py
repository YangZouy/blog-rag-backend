"""Conversation-aware query preparation for follow-up questions.

This module deliberately keeps the fast path deterministic: an independent
question is returned unchanged and does not invoke an LLM.
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

from api.models import ConversationTurn
from core.llm import get_classify_llm

logger = logging.getLogger("blog-rag")

_REFERENCE_MARKERS = re.compile(
    r"(?:^|[，。！？、\s])(?:那|那么|然后|所以|它|这个|这个方案|这个项目|该方案|上述|前者|后者|其中|还要|还需要|继续|为什么还|怎么还|是否也)",
    re.IGNORECASE,
)
_REWRITE_SYSTEM = (
    "你是博客问答的检索问题改写器。把用户当前追问改写成一个独立、明确的中文检索问题。"
    "只补全指代对象和必要上下文，不回答问题，不添加历史中没有的事实。"
    "只输出改写后的单句问题。"
)


def _normalise_history(history: Sequence[ConversationTurn] | None) -> list[ConversationTurn]:
    """Keep a small, bounded context while preserving conversation order."""
    if not history:
        return []
    # The API already caps history at two turns. Keep this guard for internal callers.
    return [turn for turn in history[-4:] if turn.content.strip()][-2:]


def is_referential_follow_up(query: str, history: Sequence[ConversationTurn] | None) -> bool:
    """Return true only when a query contains a likely anaphoric reference."""
    return bool(_normalise_history(history)) and bool(_REFERENCE_MARKERS.search(query.strip()))


def _history_text(history: Sequence[ConversationTurn]) -> str:
    return "\n".join(f"- {turn.content.strip()[:500]}" for turn in history)


def prepare_query(
    query: str,
    history: Sequence[ConversationTurn] | None = None,
) -> tuple[str, bool]:
    """Return ``(retrieval_query, rewritten)`` for the current turn."""
    current = query.strip()
    context = _normalise_history(history)
    if not is_referential_follow_up(current, context):
        return current, False

    prompt = (
        f"最近的用户问题：\n{_history_text(context)}\n\n"
        f"当前用户追问：\n{current}\n\n"
        "输出独立检索问题："
    )
    try:
        response = get_classify_llm().invoke(
            [{"role": "system", "content": _REWRITE_SYSTEM}, {"role": "user", "content": prompt}]
        )
        rewritten = re.sub(r"\s+", " ", (response.content or "").strip()).strip('"“”')
        if rewritten and len(rewritten) <= 200:
            return rewritten, True
    except Exception:  # noqa: BLE001 - rewrite is optional and must not break retrieval
        logger.warning("query rewrite failed; using original query", exc_info=True)
    return current, False
