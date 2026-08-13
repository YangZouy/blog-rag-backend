"""RAG pipeline: 前置意图路由 → (hybrid+rerank 检索) → 生成。

在检索之前先判断意图（问候/闲聊/实时数据直接短路，不浪费检索）；
SSE 新增 stage 事件，让前端显示真实进度而非假定时器。
"""
from __future__ import annotations

import json
import logging
import os
import re
import concurrent.futures
import time
from dataclasses import replace
from functools import lru_cache
from typing import Iterator, List, Mapping

from api.models import Citation, ConversationTurn, RunTrace, SearchResponse, TraceEvent
from api.retriever import retrieve_with_rerank
from core.config import get_settings
from core.conversation import prepare_query
from core.evidence import EvidenceAssessment, EvidenceStatus, assess_evidence
from core.intent import Intent, rule_intent, classify_intent
from core.llm import get_gen_llm
from core.planning import RetrievalPlan, build_retrieval_plan
from core.remediation import RemedyDecision, choose_remedy
from core.rerank import rerank
from core.observability import timed_stage
from data.parse_hexo import DocumentChunk

logger = logging.getLogger("blog-rag")

_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "blog_index.json")
)

_LIVE_DATA_ANSWER = (
    "我不能提供实时天气、价格、新闻或路况，也没有可调用的外部工具。"
    "建议使用天气、地图或新闻应用查询最新信息。"
)

_GENERATE_PROMPT = """你是博主邹阳博客的 AI 问答助手。请根据下面提供的「博客资料」回答用户问题。

## 回答要求（严格）
1. **先给结论，再补充关键要点**；不要铺陈背景、不要机械罗列
2. **严格控制在 500 字以内**；资料再多也只保留最核心、最相关的要点，技术类问题可适当展开步骤
3. **不要以"根据检索到的资料""根据博客资料"等套话开头**，直接回答用户问题
4. 用 Markdown 组织：分段清晰，可用 **加粗** 和 `代码` 突出关键信息，不要使用一级/二级大标题
5. **不要在答案里写 [1][2][3] 这类引用编号**——前端会单独展示「推荐阅读」列表
6. 博客资料是关于本站事实的唯一依据。只有资料明确写到这个博客、博主或对应项目时，才能归因给本站；
   不要把其他项目、学习笔记或案例中的技术栈混成本站 AI 问答的实现。
7. 资料不足或只有主题相近的资料时，明确说明没有检索到可确认的本站资料，不要猜测博主使用过的平台、技术栈或部署方式。
8. 你没有实时数据和外部工具。不要输出工具调用、函数调用、CALL 指令或假想的操作步骤。
9. **意图对齐校验**：如果博客资料的侧重点与用户问题的关注点不一致，
   请主动点明「站内资料没有直接覆盖你关注的[具体方面]」，不要强行拼凑答非所问的答案。
10. **禁止在答案中输出任何 URL 或 Markdown 链接**（如 https://... 或 [文字](链接)）。推荐阅读由前端单独展示，你不要自己编造或复述链接。

## 本站文章概览（用于在资料不足时判断站内是否覆盖该话题，勿逐条复述）
{overview}

## 博客资料
{context}

## 用户问题
{query}

## 回答（简洁、500字以内）"""

_FREE_ANSWER_PROMPT = """你是博主邹阳博客的 AI 问答助手。当前没有检索到相关的博客内容，请直接基于你自己的知识回答用户问题。

## 回答要求（严格）
1. 用自然、连贯的中文回答，不要铺陈背景
2. **严格控制在 500 字以内**
3. **不要以"根据检索到的资料"等套话开头**，直接回答
4. 不要写 [1][2][3] 这类引用编号
5. 你没有实时数据和外部工具；不要输出工具调用、函数调用或 CALL 指令
6. 如果用户问的是博客本身相关但你不知道的内容，可以诚实说明你不确定
7. 用 Markdown 组织内容，不要使用大标题
8. 如果用户问的是本博客相关内容但你没有找到对应资料，可以结合下方「本站文章概览」
   诚实说明「站内暂无关于[具体话题]的文章」，并可顺带提一句站内已有的相近主题，不要编造不存在的文章或链接。
9. 不要输出任何 URL 或链接，推荐阅读由前端展示。

## 本站文章概览（用于判断站内是否覆盖该话题，勿逐条复述）
{overview}

## 用户问题
{query}

## 回答（简洁、500字以内）"""

_CHAT_PROMPT = """你是「邹阳博客」的 AI 问答小助手。用户现在不是在问博客具体内容，而是在和你打招呼或闲聊。

请用自然、简短、友好的中文回应（1-3 句）：
- 可以顺带说明你能基于博客文章回答技术/项目类问题
- 不要使用大标题，不要写 [1][2] 这类引用编号
- 不要输出工具调用、函数调用或 CALL 指令
- 如果对方问你是什么，说明你是这个博客的 AI 问答助手即可

用户说：{query}

回答："""


# ---------------- 意图路由 + 检索调度 ----------------
def _route_and_retrieve(
    plan: RetrievalPlan, top_k: int, candidate_k: int | None = None,
) -> tuple[Intent, List[DocumentChunk], Mapping[str, List[DocumentChunk]], str]:
    """前置意图路由：先规则快路短路，模糊问题再让 LLM 分类 ∥ 检索 并行。"""
    # 基于规则判定
    query = plan.queries[0]
    ruled = rule_intent(query)
    # 实时数据
    if ruled == Intent.LIVE:
        return Intent.LIVE, [], {}, "out_of_scope"
    # 问候/闲聊
    if ruled == Intent.CHAT:
        return Intent.CHAT, [], {}, "chat"

    # 模糊：LLM 分类 与 检索 并行，互不等待
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        # 一条worker判别意图
        f_cls = ex.submit(classify_intent, query)
        # 一条worker开始检索（候选数固定为 RETRIEVAL_CANDIDATE_K，见 _retrieve_docs）
        f_ret = ex.submit(_retrieve_plan, plan, candidate_k)
        intent = f_cls.result()
        retrieved, chunks_by_query = f_ret.result()
    if intent == Intent.CHAT:
        return Intent.CHAT, [], {}, "chat"   # 判为闲聊，丢弃检索结果
    return Intent.RAG, retrieved, chunks_by_query, "rag"

# 对retrieve_with_rerank进行生产安全包装
# 1、异常兜底：向量检索/embedding API超时，rerank加载失败时
# 因为包裹了try catch，所以返回空列表，上层走到free-answer
# 2、观测性：timed_stage把检索耗时打点到core.observability

def _retrieve_docs(query: str, candidate_k: int | None = None) -> List[DocumentChunk]:
    """执行 hybrid + rerank 检索，返回重排后的候选 chunk 列表（固定取 RETRIEVAL_CANDIDATE_K=8 个）。"""
    candidate_k = candidate_k or get_settings().RETRIEVAL_CANDIDATE_K
    try:
        with timed_stage("retrieve", query=query):
            return retrieve_with_rerank(query, top_k=candidate_k)
    except Exception:
        logger.exception("retrieve failed; returning empty")
        return []


def _retrieve_plan(
    plan: RetrievalPlan, candidate_k: int | None = None,
) -> tuple[List[DocumentChunk], Mapping[str, List[DocumentChunk]]]:
    """Run bounded sub-query retrieval in parallel and merge duplicate chunks."""
    def retrieve_one(query: str) -> List[DocumentChunk]:
        return _retrieve_docs(query) if candidate_k is None else _retrieve_docs(query, candidate_k)

    if len(plan.queries) == 1:
        chunks = retrieve_one(plan.queries[0])
        return chunks, {plan.queries[0]: chunks}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan.queries)) as executor:
        groups = list(executor.map(retrieve_one, plan.queries))
    # The final rerank below mutates chunk.score in place. Preserve each
    # sub-query's own score for coverage assessment before merging candidates.
    chunks_by_query = {
        query: [replace(chunk) for chunk in chunks]
        for query, chunks in zip(plan.queries, groups)
    }
    merged: dict[tuple[str, int], DocumentChunk] = {}
    for chunks in groups:
        for chunk in chunks:
            key = (chunk.slug, chunk.chunk_index)
            if key not in merged or (chunk.score or 0.0) > (merged[key].score or 0.0):
                merged[key] = chunk
    candidates = list(merged.values())
    # Per-sub-query scores are only locally comparable. Score the merged pool once
    # against the original complex question before choosing generation context.
    limit = candidate_k or get_settings().RETRIEVAL_CANDIDATE_K
    with timed_stage("rerank_merged", query=plan.original_query, candidate_k=len(candidates), return_k=limit):
        return rerank(plan.original_query, candidates, limit), chunks_by_query


def _retrieve_with_evidence(
    plan: RetrievalPlan, top_k: int,
) -> tuple[Intent, List[DocumentChunk], Mapping[str, List[DocumentChunk]], list[EvidenceAssessment], RemedyDecision | None]:
    """Run at most two retrieval rounds; only code can authorize the retry."""
    settings = get_settings()
    candidate_k = settings.RETRIEVAL_CANDIDATE_K
    intent, retrieved, chunks_by_query, _ = _route_and_retrieve(plan, top_k, candidate_k)
    if intent != Intent.RAG:
        return intent, retrieved, chunks_by_query, [], None

    assessment = assess_evidence(plan, retrieved, chunks_by_query, settings.EVIDENCE_RELEVANCE_THRESHOLD)
    if assessment.status is not EvidenceStatus.INSUFFICIENT or settings.MAX_RETRIEVAL_ROUNDS < 2:
        return intent, retrieved, chunks_by_query, [assessment], None

    remedy = choose_remedy(candidate_k, settings.RETRIEVAL_REMEDY_CANDIDATE_K)
    if remedy is None:
        return intent, retrieved, chunks_by_query, [assessment], None

    intent, retrieved, chunks_by_query, _ = _route_and_retrieve(plan, top_k, remedy.candidate_k)
    if intent != Intent.RAG:
        return intent, retrieved, chunks_by_query, [assessment], remedy
    assessment = assess_evidence(plan, retrieved, chunks_by_query, settings.EVIDENCE_RELEVANCE_THRESHOLD)
    return intent, retrieved, chunks_by_query, [EvidenceAssessment(EvidenceStatus.INSUFFICIENT), assessment], remedy

@lru_cache(maxsize=1)
def _blog_overview() -> str:
    """读取data/blog-index.json文件中的内容tags + title"""
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
        lines.append(f"- {a['title']}{tag_part}")
    return "本站已收录文章（仅用于判断站内是否覆盖某话题，勿复述）：\n" + "\n".join(lines)

def _filter_relevant_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """判断最相关chunk rerank后的分数， <0.3 全部删除 否则全部保留"""
    if not chunks:
        return []
    threshold = get_settings().RERANK_RELEVANCE_THRESHOLD
    top_score = chunks[0].score
    if top_score is None or top_score < threshold:
        return []
    return chunks


def _live_data_response() -> SearchResponse:
    return SearchResponse(answer=_LIVE_DATA_ANSWER, citations=[], fallback=False, mode="out_of_scope")


def _chat_response(query: str) -> SearchResponse:
    prompt = _CHAT_PROMPT.format(query=query)
    try:
        answer = get_gen_llm().invoke(prompt).content
    except Exception:
        answer = "你好！我是这个博客的 AI 问答助手，可以基于博客里的文章回答你的技术问题～"
    return SearchResponse(answer=answer, citations=[], fallback=False, mode="chat")


def _dedupe_citations(chunks: List[DocumentChunk], query: str, max_citations: int = 3) -> List[Citation]:
    threshold = get_settings().CITATION_MIN_SCORE
    seen: set[str] = set()
    citations: List[Citation] = []
    for i, chunk in enumerate(chunks):
        if not chunk.url or chunk.slug in seen:
            continue
        # 保底：第 1 条（rerank 最高分）无论分数都保留，避免"有答案却没链接"
        if i > 0 and chunk.score is not None and chunk.score < threshold:
            continue
        seen.add(chunk.slug)
        citations.append(Citation(
            title=chunk.title, url=chunk.url, path=chunk.path,
            snippet=(chunk.content or "")[:200],
            source=chunk.doc_type, score=round(float(chunk.score or 0.0), 4),
        ))
        if len(citations) >= max_citations:
            break
    return citations

def _select_generation_context(chunks: List[DocumentChunk], max_chunks: int) -> List[DocumentChunk]:
    """取前N个chunk"""
    return chunks[:max_chunks]


def _build_prompt(docs: List[DocumentChunk], query: str, assessment: EvidenceAssessment) -> tuple[str, str]:
    overview = _blog_overview()
    if assessment.status is EvidenceStatus.PARTIAL:
        missing = "；".join(assessment.missing_aspects)
        prefix = f"站内资料未直接覆盖以下方面：{missing}。只回答有证据支持的部分，并明确说明上述缺口。\n\n"
    else:
        prefix = ""
    context = "\n\n".join(f"### 《{d.title}》\n{d.content}" for d in docs)
    return prefix + _GENERATE_PROMPT.format(context=context, query=query, overview=overview), "rag"


def _insufficient_evidence_response(assessment: EvidenceAssessment) -> SearchResponse:
    detail = ""
    if assessment.missing_aspects:
        detail = "以下关注点缺少可靠站内证据：" + "；".join(assessment.missing_aspects) + "。"
    return SearchResponse(
        answer="站内资料不足以可靠回答这个问题。" + detail + "请补充具体文章、项目名称或希望确认的范围。",
        citations=[], fallback=False, mode="not_found", evidence_status=assessment.status.value,
        missing_aspects=list(assessment.missing_aspects),
    )


def _make_trace(
    *, plan: RetrievalPlan | None, rewritten: bool, evidence_history: list[EvidenceAssessment],
    remedy: RemedyDecision | None, decision: str, started: float, question_type: str | None = None,
) -> RunTrace:
    kind = question_type or (plan.query_type if plan else "simple")
    sub_query_count = len(plan.queries) if plan else 0
    events = [TraceEvent(name="classify_question", data={"question_type": kind})]
    if rewritten:
        events.append(TraceEvent(name="rewrite_query", data={"applied": True}))
    if plan and plan.is_complex:
        events.append(TraceEvent(name="decompose", data={"sub_query_count": sub_query_count}))
    for round_index, assessment in enumerate(evidence_history, start=1):
        events.append(TraceEvent(name="assess_evidence", data={"round": round_index, "status": assessment.status.value}))
    if remedy is not None:
        events.append(TraceEvent(name="remedy", data={"action": remedy.action.value, "candidate_k": remedy.candidate_k}))
    events.append(TraceEvent(name="finalize", data={"decision": decision}))
    return RunTrace(
        question_type=kind, rewritten=rewritten, sub_query_count=sub_query_count,
        sub_queries=list(plan.queries) if plan else [],
        retrieval_rounds=len(evidence_history), evidence_statuses=[item.status.value for item in evidence_history],
        remedy_action=remedy.action.value if remedy else None, final_decision=decision,
        total_latency_ms=round((time.perf_counter() - started) * 1000, 1), events=events,
    )

# ---------------- 对外接口 ----------------
def run_rag_with_trace(
    query: str,
    top_k: int = 5,
    history: List[ConversationTurn] | None = None,
) -> tuple[SearchResponse, List[DocumentChunk]]:
    """运行生产 RAG 链路，并返回实际送入生成模型的上下文供离线评估。"""
    started = time.perf_counter()
    retrieval_query, rewritten = prepare_query(query, history)
    plan = build_retrieval_plan(retrieval_query)
    intent, retrieved, chunks_by_query, evidence_history, remedy = _retrieve_with_evidence(plan, top_k)
    if intent == Intent.LIVE:
        response = _live_data_response()
        response.trace = _make_trace(plan=None, rewritten=False, evidence_history=[], remedy=None, decision="out_of_scope", started=started, question_type="live")
        return response, []
    if intent == Intent.CHAT:
        response = _chat_response(query)
        response.trace = _make_trace(plan=None, rewritten=False, evidence_history=[], remedy=None, decision="chat", started=started, question_type="chat")
        return response, []

    assessment = evidence_history[-1]
    if assessment.status is EvidenceStatus.INSUFFICIENT:
        response = _insufficient_evidence_response(assessment)
        response.trace = _make_trace(plan=plan, rewritten=rewritten, evidence_history=evidence_history, remedy=remedy, decision="refuse", started=started)
        return response, []
    retrieved = _filter_relevant_chunks(retrieved)
    docs = _select_generation_context(retrieved, get_settings().GENERATION_CONTEXT_K)
    citations = _dedupe_citations(docs, query)
    prompt, mode = _build_prompt(docs, query, assessment)
    try:
        with timed_stage("generate", count=len(docs)):
            answer = get_gen_llm().invoke(prompt).content
    except Exception:
        logger.exception("generation failed")
        response = SearchResponse(answer="回答生成失败，请稍后重试。", citations=citations, fallback=True, mode="error")
        response.trace = _make_trace(plan=plan, rewritten=rewritten, evidence_history=evidence_history, remedy=remedy, decision="error", started=started)
        return response, docs
    response = SearchResponse(
        answer=answer, citations=citations, fallback=False, mode=mode,
        evidence_status=assessment.status.value, missing_aspects=list(assessment.missing_aspects),
    )
    response.trace = _make_trace(
        plan=plan, rewritten=rewritten, evidence_history=evidence_history, remedy=remedy,
        decision="partial_answer" if assessment.status is EvidenceStatus.PARTIAL else "answer", started=started,
    )
    return response, docs


def run_rag(query: str, top_k: int = 5, history: List[ConversationTurn] | None = None) -> SearchResponse:
    """非流式：路由 → 检索 → 一次性生成。"""
    response, _ = run_rag_with_trace(query, top_k, history)
    return response

def stream_rag(query: str, top_k: int = 5, history: List[ConversationTurn] | None = None) -> Iterator[tuple[str, dict]]:
    """流式：先回传 stage（真实阶段）→ sources（推荐阅读）→ 逐 token 答案。"""
    started = time.perf_counter()
    yield "stage", {"stage": "routing"}

    retrieval_query, rewritten = prepare_query(query, history)
    if rewritten:
        yield "stage", {"stage": "rewriting"}
    plan = build_retrieval_plan(retrieval_query)
    if plan.is_complex:
        yield "stage", {"stage": "decomposing", "sub_query_count": len(plan.queries)}
    intent, retrieved, chunks_by_query, evidence_history, remedy = _retrieve_with_evidence(plan, top_k)

    if intent == Intent.LIVE:
        resp = _live_data_response()
        resp.trace = _make_trace(plan=None, rewritten=False, evidence_history=[], remedy=None, decision="out_of_scope", started=started, question_type="live")
        yield "sources", {"citations": [], "mode": resp.mode}
        yield "token", {"text": resp.answer}
        yield "trace", resp.trace.model_dump()
        yield "done", resp.model_dump()
        return

    if intent == Intent.CHAT:
        # stage事件：routing retrieving generating
        yield "stage", {"stage": "generating"}
        yield "sources", {"citations": [], "mode": "chat"}
        parts: list[str] = []
        try:
            # model.stream逐token输出
            for chunk in get_gen_llm().stream(_CHAT_PROMPT.format(query=query)):
                token = getattr(chunk, "content", "") or ""
                if token:
                    parts.append(token)
                    yield "token", {"text": token}
        except Exception:
            yield "error", {"message": "回答生成失败，请稍后重试。"}
        response = SearchResponse(answer="".join(parts), citations=[], fallback=False, mode="chat")
        response.trace = _make_trace(plan=None, rewritten=False, evidence_history=[], remedy=None, decision="chat", started=started, question_type="chat")
        yield "trace", response.trace.model_dump()
        yield "done", response.model_dump()
        return

    # ---- RAG 路径 ----
    yield "stage", {"stage": "retrieving"}
    if remedy is not None:
        yield "stage", {"stage": "remedy", "action": remedy.action.value, "candidate_k": remedy.candidate_k}
    assessment = evidence_history[-1]
    if assessment.status is EvidenceStatus.INSUFFICIENT:
        response = _insufficient_evidence_response(assessment)
        response.trace = _make_trace(plan=plan, rewritten=rewritten, evidence_history=evidence_history, remedy=remedy, decision="refuse", started=started)
        yield "sources", {"citations": [], "mode": response.mode}
        yield "token", {"text": response.answer}
        yield "trace", response.trace.model_dump()
        yield "done", response.model_dump()
        return
    retrieved = _filter_relevant_chunks(retrieved)
    docs = _select_generation_context(retrieved, get_settings().GENERATION_CONTEXT_K)
    citations = _dedupe_citations(docs, query)
    prompt, mode = _build_prompt(docs, query, assessment)

    yield "stage", {"stage": "generating"}
    yield "sources", {"citations": [c.model_dump() for c in citations], "mode": mode}

    parts = []
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
        response = SearchResponse(answer="回答生成失败，请稍后重试。", citations=citations, fallback=True, mode="error")
        response.trace = _make_trace(plan=plan, rewritten=rewritten, evidence_history=evidence_history, remedy=remedy, decision="error", started=started)
        yield "trace", response.trace.model_dump()
        yield "done", response.model_dump()
        return

    response = SearchResponse(
        answer="".join(parts), citations=citations, fallback=False, mode=mode,
        evidence_status=assessment.status.value, missing_aspects=list(assessment.missing_aspects),
    )
    response.trace = _make_trace(
        plan=plan, rewritten=rewritten, evidence_history=evidence_history, remedy=remedy,
        decision="partial_answer" if assessment.status is EvidenceStatus.PARTIAL else "answer", started=started,
    )
    yield "trace", response.trace.model_dump()
    yield "done", response.model_dump()
