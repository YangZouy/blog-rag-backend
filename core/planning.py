"""Bounded planning for complex blog-RAG questions.

The model may suggest retrieval sub-queries, but code owns the trigger,
maximum count, deduplication and fallback behaviour.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from core.llm import get_planning_llm

logger = logging.getLogger("blog-rag")

MAX_SUB_QUERIES = 4
_COMPLEX_MARKERS = re.compile(
    r"(?:对比|比较|区别|差异|分别|各自|架构.*(?:稳定性|评测)|(?:架构|稳定性|评测).*(?:架构|稳定性|评测)|"
    r"(?:以及|和|与).*(?:架构|稳定性|评测|实现|优缺点).*(?:以及|和|与))"
)
_DECOMPOSE_SYSTEM = f"""你是博客问答的检索规划器。用户的问题已被程序判定为多目标复杂问题。
将问题拆成 2 到 {MAX_SUB_QUERIES} 个彼此不同、可独立检索的中文子问题，覆盖用户明确提出的目标。
不要回答问题，不要添加问题中没有的目标。只输出 JSON：{{"sub_queries":["..."]}}。"""


@dataclass(frozen=True)
class RetrievalPlan:
    query_type: str
    original_query: str
    queries: tuple[str, ...]

    @property
    def is_complex(self) -> bool:
        return self.query_type == "complex"


def is_obviously_complex(query: str) -> bool:
    """Cheap gate that keeps simple questions away from the planning model."""
    return bool(_COMPLEX_MARKERS.search(query.strip()))


def _parse_sub_queries(content: str, original: str) -> tuple[str, ...]:
    try:
        payload = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
        items = payload.get("sub_queries", []) if isinstance(payload, dict) else []
    except (json.JSONDecodeError, AttributeError):
        return ()

    seen = {original.strip()}
    result: list[str] = []
    for item in items:
        candidate = re.sub(r"\s+", " ", str(item).strip())
        if not candidate or len(candidate) > 200 or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
        if len(result) == MAX_SUB_QUERIES:
            break
    return tuple(result) if len(result) >= 2 else ()


def build_retrieval_plan(query: str) -> RetrievalPlan:
    """Create a bounded retrieval plan, falling back to one original query."""
    original = query.strip()
    if not is_obviously_complex(original):
        return RetrievalPlan(query_type="simple", original_query=original, queries=(original,))
    try:
        response = get_planning_llm().invoke(
            [{"role": "system", "content": _DECOMPOSE_SYSTEM}, {"role": "user", "content": original}]
        )
        sub_queries = _parse_sub_queries(response.content or "", original)
        if sub_queries:
            return RetrievalPlan(query_type="complex", original_query=original, queries=sub_queries)
    except Exception:  # noqa: BLE001 - planning is optional; preserve the stable path
        logger.warning("complex query planning failed; using original query", exc_info=True)
    return RetrievalPlan(query_type="simple", original_query=original, queries=(original,))
