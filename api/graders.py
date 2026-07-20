"""Relevance grading, query rewriting, and routing decisions."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import List

from core.config import get_settings
from core.llm import get_grade_llm
from data.parse_hexo import DocumentChunk


_GRADE_PROMPT = """你是一个严格的相关性判定员。判断下面的文档片段能否为用户问题提供直接、具体的答案。
用户问题：{query}
文档标题：{title}
文档标签：{tags}
文档章节：{section}
文档片段：{content}
只能回答 yes 或 no："""

_REWRITE_PROMPT = """用户的问题未能从博客检索到的片段中得到答案。请把原问题改写成更精准、更适合博客检索的查询语句；如果是多跳/复合问题，请拆成若干子问题。
只输出改写后的查询（多个子问题用 '; ' 连接），不要解释。
原问题：{query}
检索到的（可能无关的）片段提示：{hints}
"""

_DECISION_PROMPT = """你是一个博客问答系统的检索路由决策器。根据用户查询，以及从博客中检索到（并经相关性筛选）的文档片段，决定下一步动作。

用户查询：{query}

检索到的博客文档（最多 {k} 篇，已按相关性排序）：
{docs}

请严格从以下三个动作中选择一个，并以 JSON 返回：{{"action": "ANSWER"|"REWRITE"|"WEB", "reason": "简短理由"}}
- ANSWER：检索到的博客文档已包含足够具体的信息来回答该问题，可直接生成答案。
- REWRITE：博客很可能涵盖该主题，但当前查询措辞不佳/有歧义/是多跳问题，改写查询后可能检索到更好的文档。（仅当文档部分相关或查询明显含糊/复合时才选）
- WEB：查询明显超出本博客范围（最新资讯、外部产品、实时数据、博客不可能知道的内容），需要联网搜索。

若检索文档为空，则在 REWRITE（博客可能涵盖）与 WEB（明显外部）之间选择。
"""


def grade_documents(docs: List[DocumentChunk], query: str) -> List[DocumentChunk]:
    if not docs:
        return []
    if all(doc.doc_type == "web" for doc in docs):
        return docs
    try:
        llm = get_grade_llm()
    except Exception:
        return docs

    def grade_one(doc: DocumentChunk) -> DocumentChunk | None:
        prompt = _GRADE_PROMPT.format(
            query=query,
            title=doc.title or "",
            tags="、".join(doc.tags or []),
            section=doc.section or "",
            content=doc.content,
        )
        try:
            response = llm.invoke(prompt).content.strip().lower()
        except Exception:
            return doc
        if response.startswith("y"):
            return doc
        return None

    max_workers = min(len(docs), max(1, get_settings().GRADE_MAX_CONCURRENCY))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rag-grade") as executor:
        graded = executor.map(grade_one, docs)
        return [doc for doc in graded if doc is not None]


def transform_query(query: str, hints: List[str] | None = None) -> str:
    """带上下文的改写：利用已检索到的（可能无关的）片段做 HyDE / 分解式重写。"""
    llm = get_grade_llm()
    hint_text = "\n".join(f"- {h}" for h in (hints or [])[:5]) or "（无）"
    prompt = _REWRITE_PROMPT.format(query=query, hints=hint_text)
    try:
        return llm.invoke(prompt).content.strip() or query
    except Exception:
        return query


def decide_action(query: str, docs: List[DocumentChunk]) -> str:
    """LLM 路由决策：返回 'answer' / 'rewrite' / 'web'（小写）。"""
    llm = get_grade_llm()
    doc_lines = []
    for i, d in enumerate(docs[:5], 1):
        snippet = (d.content or "")[:200].replace("\n", " ")
        doc_lines.append(f"[{i}] ({d.doc_type}) {d.title}\n    {snippet}")
    doc_text = "\n".join(doc_lines) or "（无相关文档）"
    prompt = _DECISION_PROMPT.format(query=query, k=len(docs[:5]), docs=doc_text)
    try:
        raw = llm.invoke(prompt).content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        if '"action"' in raw:
            try:
                return json.loads(raw).get("action", "answer").lower()
            except Exception:
                pass
        low = raw.lower()
        if "web" in low:
            return "web"
        if "rewrite" in low:
            return "rewrite"
        return "answer"
    except Exception:
        return "answer"
