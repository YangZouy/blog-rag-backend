"""Simplified RAG pipeline: hybrid retrieval + rerank -> generate.

设计原则（2026-07-20 简化）:
- 检索层（向量 + BM25 hybrid + cross-encoder rerank）已验证足够强
  （eval: R@3=0.98 / R@10=1.00 / MRR=0.87），直接进入生成即可。
- 链路压缩为两步：retrieve_with_rerank(top-k) -> generate。
- 检索为空（或没有相关 chunk）时，直接让生成模型基于自身知识自由回答。
- 答案中不出现 [n] 引用标记；前端单独展示「推荐阅读」文章列表。
"""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Iterator, List

from api.models import Citation, ConversationTurn, SearchResponse
from api.retriever import retrieve_with_rerank
from core.config import get_settings
from core.llm import get_gen_llm
from core.observability import timed_stage
from data.parse_hexo import DocumentChunk

logger = logging.getLogger("blog-rag")

# 全站文章清单（由 api/build_index.py 生成）。让 LLM 在检索为空时也能判断
# 「站内有什么 / 没有什么」，减少幻觉与错误推荐。详见 P3 优化说明。
_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "blog_index.json")
)

_LIVE_DATA_PATTERNS = (
    r"(?:今天|现在|当前|实时|最新).{0,8}(?:天气|气温|温度|降雨|空气质量|股价|汇率|金价|油价|新闻|路况)",
    r"(?:天气|气温|温度|降雨|空气质量).{0,8}(?:怎么样|如何|多少|查询|预报|吗)",
)

_LIVE_DATA_ANSWER = (
    "我不能提供实时天气、价格、新闻或路况，也没有可调用的外部工具。"
    "建议使用天气、地图或新闻应用查询最新信息。"
)

# 有检索结果时：基于博客资料回答，但允许模型在资料不相关时自由发挥
_GENERATE_PROMPT = """你是博主邹阳博客的 AI 问答助手。请根据下面提供的「博客资料」回答用户问题。

## 回答要求（严格）
1. **先给结论，再补充关键要点**；不要铺陈背景、不要机械罗列
2. **严格控制在 300 字以内**；资料再多也只保留最核心、最相关的 3-5 个要点
3. **不要以"根据检索到的资料""根据博客资料"等套话开头**，直接回答用户问题
4. 用 Markdown 组织：分段清晰，可用 **加粗** 和 `代码` 突出关键信息，不要使用一级/二级大标题
5. **不要在答案里写 [1][2][3] 这类引用编号**——前端会单独展示「推荐阅读」列表
6. 博客资料是关于本站事实的唯一依据。只有资料明确写到这个博客、博主或对应项目时，才能归因给本站；
   不要把其他项目、学习笔记或案例中的技术栈混成本站 AI 问答的实现。
7. 资料不足或只有主题相近的资料时，明确说明没有检索到可确认的本站资料，不要猜测博主使用过的平台、技术栈或部署方式。
8. 你没有实时数据和外部工具。不要输出工具调用、函数调用、CALL 指令或假想的操作步骤。
9. **意图对齐校验**：如果博客资料的侧重点与用户问题的关注点不一致
   （例如用户问「技术实现 / 架构方案」，但资料只讲了「使用方法 / 数据录入方式」；
   或用户问「A 方案的优劣」，但资料只介绍了 B 方案），
   请主动点明「站内资料没有直接覆盖你关注的[具体方面]」，
   不要强行从侧面资料里拼凑出一个看似相关、实则答非所问的答案。

## 本站文章概览（用于在资料不足时判断站内是否覆盖该话题，勿逐条复述）
{overview}

## 博客资料
{context}

## 用户问题
{query}

## 本轮之前的用户问题
{history}

## 回答（简洁、300字以内）"""

# 检索为空时：没有资料可用，直接让模型自由回答
_FREE_ANSWER_PROMPT = """你是博主邹阳博客的 AI 问答助手。当前没有检索到相关的博客内容，请直接基于你自己的知识回答用户问题。

## 回答要求（严格）
1. 用自然、连贯的中文回答，不要铺陈背景
2. **严格控制在 300 字以内**
3. **不要以"根据检索到的资料"等套话开头**，直接回答
4. 不要写 [1][2][3] 这类引用编号
5. 你没有实时数据和外部工具；不要输出工具调用、函数调用或 CALL 指令
6. 如果用户问的是博客本身相关但你不知道的内容，可以诚实说明你不确定
7. 用 Markdown 组织内容，不要使用大标题
8. 如果用户问的是本博客相关内容但你没有找到对应资料，可以结合下方「本站文章概览」
   诚实说明「站内暂无关于[具体话题]的文章」，并可顺带提一句站内已有的相近主题，
   不要编造不存在的文章或链接。

## 本站文章概览（用于判断站内是否覆盖该话题，勿逐条复述）
{overview}

## 用户问题
{query}

## 回答（简洁、300字以内）"""


def _retrieve_docs(query: str, top_k: int) -> List[DocumentChunk]:
    """执行 hybrid + rerank 检索，返回重排后的候选 chunk 列表。"""
    candidate_k = max(top_k, get_settings().RETRIEVAL_CANDIDATE_K)
    try:
        with timed_stage("retrieve", query=query):
            return retrieve_with_rerank(query, top_k=candidate_k)
    except Exception:
        logger.exception("retrieve failed; returning empty")
        return []


def _build_retrieval_query(query: str, history: List[ConversationTurn]) -> str:
    """将短追问补成可检索的独立问题，不使用模型改写。"""
    previous = [turn.content.strip() for turn in history if turn.content.strip()]
    if not previous:
        return query
    return f"上一轮问题：{previous[-1]}\n当前追问：{query}"


def _format_history(history: List[ConversationTurn]) -> str:
    if not history:
        return "无"
    return "\n".join(f"- {turn.content.strip()}" for turn in history if turn.content.strip()) or "无"


@lru_cache(maxsize=1)
def _blog_overview() -> str:
    """把全站文章清单拼成紧凑的 Markdown 列表，注入 prompt。

    仅在 build_index.py 已生成 data/blog_index.json 时生效；否则返回空串，
    对现有链路零侵入。列表只放标题 + 链接 + 标签，控制在可接受 token 量级。
    """
    if not os.path.exists(_INDEX_PATH):
        return ""
    try:
        with open(_INDEX_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        articles = data.get("articles", [])
    except Exception:
        logger.warning("blog_index.json 读取失败，跳过全站概览注入")
        return ""
    if not articles:
        return ""
    lines = []
    for a in articles:
        tags = a.get("tags") or []
        tag_part = f" 〔{'、'.join(tags)}〕" if tags else ""
        lines.append(f"- [{a['title']}]({a['url']}){tag_part}")
    return "本站已收录文章（标题可点链接）：\n" + "\n".join(lines)


def _filter_relevant_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """无可信站内命中时，不把语义相近但无关的文章交给生成或前端。"""
    if not chunks:
        return []
    threshold = get_settings().RERANK_RELEVANCE_THRESHOLD
    top_score = chunks[0].score
    if top_score is None or top_score < threshold:
        return []
    return chunks


def _is_live_data_query(query: str) -> bool:
    """识别不适合站内 RAG 的实时信息请求。"""
    text = (query or "").strip().lower()
    return any(re.search(pattern, text) for pattern in _LIVE_DATA_PATTERNS)


def _live_data_response() -> SearchResponse:
    return SearchResponse(
        answer=_LIVE_DATA_ANSWER,
        citations=[],
        fallback=False,
        mode="out_of_scope",
    )


def _dedupe_citations(
    chunks: List[DocumentChunk], query: str, max_citations: int = 2
) -> List[Citation]:
    """按文章（slug）去重，返回「推荐阅读」列表。

    - 完全按 rerank 后的相关性排序取前 max_citations 篇，不固定 3 篇。
    - 不再对「关于我」做特殊置顶：博主信息已前置到前端开场白，
      避免每个回答都硬塞关于我链接。
    - 同一篇文章会被切成多个 chunk，但前端只需要展示一篇文章链接。
    - **置信度门槛（CITATION_MIN_SCORE）高于上下文门槛**：只有 rerank 分数
      足够高的文章才进推荐阅读。宁可少推、不推，也比推一堆基本不相关的
      文章体验更好（见 P0 优化说明）。
    """
    threshold = get_settings().CITATION_MIN_SCORE
    seen: set[str] = set()
    citations: List[Citation] = []
    for chunk in chunks:
        if not chunk.url or chunk.slug in seen:
            continue
        # 跳过置信度不达标的文章：即使它侥幸进了候选池。
        if chunk.score is not None and chunk.score < threshold:
            continue
        seen.add(chunk.slug)
        citations.append(
            Citation(
                title=chunk.title,
                url=chunk.url,
                snippet=(chunk.content or "")[:200],
                source=chunk.doc_type,
                score=round(float(chunk.score or 0.0), 4),
            )
        )
        if len(citations) >= max_citations:
            break
    return citations


def _select_generation_context(
    chunks: List[DocumentChunk], max_chunks: int
) -> List[DocumentChunk]:
    """保留不同文章的最高分 chunk，控制生成上下文大小。"""
    selected: List[DocumentChunk] = []
    seen_slugs: set[str] = set()
    for chunk in chunks:
        # 同一篇文章的相邻块通常高度重叠，不重复消耗生成模型上下文。
        identity = chunk.slug or f"{chunk.title}:{chunk.chunk_index}"
        if identity in seen_slugs:
            continue
        seen_slugs.add(identity)
        selected.append(chunk)
        if len(selected) >= max_chunks:
            break
    return selected


def _build_prompt(
    docs: List[DocumentChunk], query: str, history: List[ConversationTurn]
) -> tuple[str, str]:
    """根据是否有检索结果选择 prompt，返回 (prompt, mode)。"""
    overview = _blog_overview()
    if not docs:
        return _FREE_ANSWER_PROMPT.format(query=query, overview=overview), "free"
    context = "\n\n".join(f"### 《{d.title}》\n{d.content}" for d in docs)
    return _GENERATE_PROMPT.format(
        context=context, query=query, history=_format_history(history), overview=overview
    ), "rag"


def run_rag(
    query: str, top_k: int = 5, history: List[ConversationTurn] | None = None
) -> SearchResponse:
    """非流式：检索 -> 一次性生成。"""
    if _is_live_data_query(query):
        return _live_data_response()
    history = history or []
    retrieved = _filter_relevant_chunks(
        _retrieve_docs(_build_retrieval_query(query, history), top_k)
    )
    context_k = min(top_k, get_settings().GENERATION_CONTEXT_K)
    docs = _select_generation_context(retrieved, context_k)
    citations = _dedupe_citations(docs, query)
    prompt, mode = _build_prompt(docs, query, history)
    try:
        with timed_stage("generate", count=len(docs)):
            answer = get_gen_llm().invoke(prompt).content
    except Exception:
        logger.exception("generation failed")
        return SearchResponse(
            answer="回答生成失败，请稍后重试。",
            citations=citations,
            fallback=True,
            mode="error",
        )
    return SearchResponse(
        answer=answer,
        citations=citations,
        fallback=False,
        mode=mode,
    )


def stream_rag(
    query: str, top_k: int = 5, history: List[ConversationTurn] | None = None
) -> Iterator[tuple[str, dict]]:
    """流式：先回传 sources（推荐阅读），再逐 token 输出答案。"""
    if _is_live_data_query(query):
        response = _live_data_response()
        yield "sources", {"citations": [], "mode": response.mode}
        yield "token", {"text": response.answer}
        yield "done", response.model_dump()
        return
    history = history or []
    retrieved = _filter_relevant_chunks(
        _retrieve_docs(_build_retrieval_query(query, history), top_k)
    )
    context_k = min(top_k, get_settings().GENERATION_CONTEXT_K)
    docs = _select_generation_context(retrieved, context_k)
    citations = _dedupe_citations(docs, query)
    prompt, mode = _build_prompt(docs, query, history)

    # 先把「推荐阅读」推给前端，数量与答案解耦，永远不会错配
    yield "sources", {"citations": [c.model_dump() for c in citations], "mode": mode}

    parts: list[str] = []
    try:
        with timed_stage("generate", count=len(docs)):
            for chunk in get_gen_llm().stream(prompt):
                token = getattr(chunk, "content", "") or ""
                if token:
                    parts.append(token)
                    yield "token", {"text": token}
    except Exception:
        logger.exception("stream generation failed")
        yield "error", {"message": "回答生成失败，请稍后重试。"}
        yield "done", SearchResponse(
            answer="回答生成失败，请稍后重试。",
            citations=citations,
            fallback=True,
            mode="error",
        ).model_dump()
        return

    yield "done", SearchResponse(
        answer="".join(parts),
        citations=citations,
        fallback=False,
        mode=mode,
    ).model_dump()
