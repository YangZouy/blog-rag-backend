"""Simplified RAG pipeline: hybrid retrieval + rerank -> generate.

设计原则（2026-07-20 简化）:
- 检索层（向量 + BM25 hybrid + cross-encoder rerank）已验证足够强
  （eval: R@3=0.98 / R@10=1.00 / MRR=0.87），直接进入生成即可。
- 链路压缩为两步：retrieve_with_rerank(top-k) -> generate。
- 检索为空（或没有相关 chunk）时，不接 web search，直接让生成模型
  基于自身知识自由回答。
- 答案中不出现 [n] 引用标记；前端单独展示「推荐阅读」文章列表。
"""
from __future__ import annotations

import logging
from typing import Iterator, List

from api.models import Citation, SearchResponse
from api.retriever import retrieve_with_rerank
from core.config import get_settings
from core.llm import get_gen_llm
from core.observability import timed_stage
from data.parse_hexo import DocumentChunk

logger = logging.getLogger("blog-rag")

# 有检索结果时：基于博客资料回答，但允许模型在资料不相关时自由发挥
_GENERATE_PROMPT = """你是博主邹阳博客的 AI 问答助手。请根据下面提供的「博客资料」回答用户问题。

## 回答要求（严格）
1. **先给结论，再补充关键要点**；不要铺陈背景、不要机械罗列
2. **严格控制在 300 字以内**；资料再多也只保留最核心、最相关的 3-5 个要点
3. **不要以"根据检索到的资料""根据博客资料"等套话开头**，直接回答用户问题
4. 用 Markdown 组织：分段清晰，可用 **加粗** 和 `代码` 突出关键信息，不要使用一级/二级大标题
5. **不要在答案里写 [1][2][3] 这类引用编号**——前端会单独展示「推荐阅读」列表
6. 如果资料与问题明显不相关，直接基于你的知识简要回答，不要硬说"博客里没有"

## 博客资料
{context}

## 用户问题
{query}

## 回答（简洁、300字以内）"""

# 检索为空时：没有资料可用，直接让模型自由回答
_FREE_ANSWER_PROMPT = """你是博主邹阳博客的 AI 问答助手。当前没有检索到相关的博客内容，请直接基于你自己的知识回答用户问题。

## 回答要求（严格）
1. 用自然、连贯的中文回答，不要铺陈背景
2. **严格控制在 300 字以内**
3. **不要以"根据检索到的资料"等套话开头**，直接回答
4. 不要写 [1][2][3] 这类引用编号
5. 如果用户问的是博客本身相关但你不知道的内容，可以诚实说明你不确定
6. 用 Markdown 组织内容，不要使用大标题

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


def _dedupe_citations(
    chunks: List[DocumentChunk], query: str, max_citations: int = 2
) -> List[Citation]:
    """按文章（slug）去重，返回「推荐阅读」列表。

    - 完全按 rerank 后的相关性排序取前 max_citations 篇，不固定 3 篇。
    - 不再对「关于我」做特殊置顶：博主信息已前置到前端开场白，
      避免每个回答都硬塞关于我链接。
    - 同一篇文章会被切成多个 chunk，但前端只需要展示一篇文章链接。
    """
    seen: set[str] = set()
    citations: List[Citation] = []
    for chunk in chunks:
        if not chunk.url or chunk.slug in seen:
            continue
        seen.add(chunk.slug)
        citations.append(
            Citation(
                title=chunk.title,
                url=chunk.url,
                snippet=(chunk.content or "")[:200],
                source=chunk.doc_type,
            )
        )
        if len(citations) >= max_citations:
            break
    return citations


def _build_prompt(docs: List[DocumentChunk], query: str) -> tuple[str, str]:
    """根据是否有检索结果选择 prompt，返回 (prompt, mode)。"""
    if not docs:
        return _FREE_ANSWER_PROMPT.format(query=query), "free"
    context = "\n\n".join(f"### 《{d.title}》\n{d.content}" for d in docs)
    return _GENERATE_PROMPT.format(context=context, query=query), "rag"


def run_rag(query: str, top_k: int = 5) -> SearchResponse:
    """非流式：检索 -> 一次性生成。"""
    docs = _retrieve_docs(query, top_k)
    citations = _dedupe_citations(docs, query)
    prompt, mode = _build_prompt(docs, query)
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


def stream_rag(query: str, top_k: int = 5) -> Iterator[tuple[str, dict]]:
    """流式：先回传 sources（推荐阅读），再逐 token 输出答案。"""
    docs = _retrieve_docs(query, top_k)
    citations = _dedupe_citations(docs, query)
    prompt, mode = _build_prompt(docs, query)

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
