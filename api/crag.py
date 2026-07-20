"""CRAG（Corrective RAG）—— Phase 1：基于 reranker 分数的 3 类评分器 + web 门控。

设计原则（与现有 rag_graph 解耦，纯加法，不修改线上图）：
- 评分器直接复用 retrieve_with_rerank 已经算出的 chunk.score（bge-reranker-base 的
  logit 分），不额外调用 LLM，零成本。
- 3 类：CORRECT / INCORRECT / AMBIGUOUS，由阈值把 logit 分切三段（阈值需经
  eval/eval_grader.py 校准，见文件顶部常量与脚本说明）。
- web 触发规则（用户定义）：
    A. top5 中 incorrect 数量 > 3 且 所有 top5 文档分数都低于相关性地板 -> KB 不足，web
    B. 存在 AMBIGUOUS 文档 且 查询是时间敏感 -> 时效缺口，web
- 知识精炼（Knowledge Refinement）Phase 1 先做"文档级"：丢弃 INCORRECT，保留
  CORRECT+AMBIGUOUS；web 路径合并 KB+web 后再 rerank。段落级（拆句留关键句）是
  Phase D 的 LLM 压缩，留 TODO 钩子。

Phase 2（D 混合，已实现）：在 AMBIGUOUS 带调用一次便宜 LLM（GRADE_MODEL, temp=0）裁决，
其余用 reranker 分秒过。由 config.CRAG_PHASE_D 开关控制（默认 True，仅在 CRAG 主路生效）。
"""
from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Dict, Iterator, List, Tuple

from api.models import Citation, SearchResponse
from api.retriever import retrieve_with_rerank
from api.web_search import web_search
from core.config import get_settings
from core.llm import get_gen_llm, get_grade_llm
from core.observability import timed_stage
from core.rerank import rerank
from data.parse_hexo import DocumentChunk

# 复用 rag_graph 里已验证的生成逻辑，避免 prompt 双份维护
from api.rag_graph import _GENERATE_PROMPT, _clamp_citations, _dedupe_citations

logger = logging.getLogger("blog-rag")

# ----------------------------------------------------------------------------
# 阈值（CALIBRATE）：bge-reranker-base 输出 logit。
# 校准依据（eval/eval_grader.py --dump，2026-07-19 实测分布）：
#   CORRECT  median=0.903 / INCORRECT median=0.035，两类中位数差 ~25x。
#   网格扫描目标 useful_acc=(1-模糊带)*doc_acc，约束 amb%<=15% 且 keepR>=0.65，
#   得最优工作点 CORRECT_THRESHOLD=0.3 / INCORRECT_THRESHOLD=0.2：
#     - 模糊带仅 4.2%（Phase D 的 LLM-as-judge 只裁决这 4.2% 文档，成本极低）
#     - 精炼精度 dropP=0.774（被判 INCORRECT 丢弃的文档中 77% 真为 junk）
#     - 保留召回 keepR=0.661（真相关文档极少被误丢）
#   -> WEB_RELEVANCE_FLOOR=0.5：top5 无任何文档 >=0.5 才算"无相关信号"。
#      注意：若设为 0.0（初版默认值），因 reranker 分几乎恒正，has_relevant 永远 True，
#      web 分支永远不触发，门控等于死代码（WEB_PROBES 5/5 触发已验证 0.3/0.2/0.5 配置）。
# ----------------------------------------------------------------------------
CORRECT_THRESHOLD = 0.3       # score >= 此值 -> CORRECT
INCORRECT_THRESHOLD = 0.2     # score <= 此值 -> INCORRECT；之间 -> AMBIGUOUS
WEB_N_INCORRECT = 3           # "超过 3 篇" => n_incorrect > 3（即 >=4）
WEB_RELEVANCE_FLOOR = 0.5     # top5 中无任何文档 >= 此分 => 无相关信号（必须为正阈值）


class GradeLabel(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"


# 时间敏感词：命中任一词即认为查询可能超出静态博客范围，需 web 兜底
_TIME_PATTERNS = [
    r"最新", r"今年", r"去年", r"今天", r"昨天", r"现在", r"目前", r"当前",
    r"实时", r"股价", r"价格", r"汇率", r"版本", r"更新", r"新闻", r"发布",
    r"release", r"latest", r"version", r"v\d", r"\b20(2[4-9]|3\d)\b",
]

# 身份意图词：查询博主/作者本人信息时，应走站内 about 页，绝不应触发联网。
# 联网返回的噪声（CSDN/知乎同名用户）只会污染 context，降低答案质量。
_IDENTITY_PATTERNS = [
    r"博主", r"作者", r"你是谁", r"你叫什么", r"主人", r"站长",
    r"自我介绍", r"关于我", r"关于.*你", r"个人简历", r"履历",
    r"谁写的", r"谁做的", r"谁建的", r"联系.*博主", r"博主.*联系",
]


def is_time_sensitive(query: str) -> bool:
    """Phase 1 用关键词启发式；Phase D 可升级为 LLM 判断。"""
    low = (query or "").lower()
    return any(re.search(p, low) for p in _TIME_PATTERNS)


def is_identity_query(query: str) -> bool:
    """判断是否为身份/博主相关查询。

    身份查询的特点：用户想知道的是「这个博客是谁的」「博主叫什么」等，
    答案完全由站内 about 页面提供，联网搜索只返回无关噪声（同名用户、
    其他平台的同名文章）。这类查询必须禁用 web 门控，且 about 页豁免仅
    在此类查询下生效（非身份查询中 about 页应正常参与评分过滤）。
    """
    low = (query or "").strip()
    # 极短查询（≤4字且不含明确关键词）不判为身份，避免误伤
    if len(low) <= 4 and not any(k in low for k in ["博主", "作者", "你"]):
        return False
    return any(re.search(p, low) for p in _IDENTITY_PATTERNS)


def _label(score: float | None, correct_t: float, incorrect_t: float) -> GradeLabel:
    if score is None:
        return GradeLabel.AMBIGUOUS
    if score >= correct_t:
        return GradeLabel.CORRECT
    if score <= incorrect_t:
        return GradeLabel.INCORRECT
    return GradeLabel.AMBIGUOUS


def classify_documents(
    docs: List[DocumentChunk],
    correct_t: float = CORRECT_THRESHOLD,
    incorrect_t: float = INCORRECT_THRESHOLD,
) -> List[Tuple[DocumentChunk, GradeLabel]]:
    """逐文档 3 分类，保留 reranker 原始顺序。"""
    return [(d, _label(d.score, correct_t, incorrect_t)) for d in docs]


def decide_web_search(
    query: str,
    classified: List[Tuple[DocumentChunk, GradeLabel]],
    n_incorrect_trigger: int = WEB_N_INCORRECT,
    relevance_floor: float = WEB_RELEVANCE_FLOOR,
) -> bool:
    """返回 True 表示需要 web 检索（KB 不足 / 时效缺口）。"""
    # P1 修复：身份查询（博主是谁/叫什么等）绝不应联网。
    # 联网只返回同名用户/无关平台噪声，污染 context 且降低答案质量。
    if is_identity_query(query):
        return False

    top = classified[:5]
    if not top:
        return True  # 啥都没捞到 -> 直接 web

    n_incorrect = sum(1 for _, lbl in top if lbl == GradeLabel.INCORRECT)
    has_relevant = any((d.score or float("-inf")) >= relevance_floor for d, _ in top)
    # top5中INCORRECT数量>3且top5里没有任何一篇的分数达到0.5
    kb_insufficient = (n_incorrect > n_incorrect_trigger) and (not has_relevant)

    time_gap = is_time_sensitive(query) and any(
        lbl == GradeLabel.AMBIGUOUS for _, lbl in top
    )
    return kb_insufficient or time_gap


def _is_about_page(d: DocumentChunk) -> bool:
    """判断是否为身份意图注入的「关于我」页面。

    About 页的 reranker 分天然偏低（cross-encoder 不擅长匹配极短 identity query 与
    长篇自我介绍文本），但这类文档在命中身份 token 时是**显式召回**的语义保证，
    不应被评分器误判为 INCORRECT 而丢掉。豁免规则：slug 含 'about' 或标题含
    「关于」/「自我介绍」。
    """
    slug = (d.slug or "").lower()
    title = (d.title or "")
    return "about" in slug or "关于" in title or "自我介绍" in title


def _keep_context(
    classified: List[Tuple[DocumentChunk, GradeLabel]], top_k: int,
    is_identity: bool = False,  # P2 修复：仅在身份查询时豁免 about 页
) -> List[DocumentChunk]:
    """知识精炼（文档级）：丢弃 INCORRECT，保留 CORRECT+AMBIGUOUS，维持 reranker 顺序。

    About 页（身份意图注入）**仅当查询是身份类时**才豁免：
    - 身份查询（"博主叫什么"）：about 页是唯一正确答案源，即使分低也保留
    - 非身份查询（"吴恩达课程"/"TypeScript教程"）：about 页与问题无关，
      reranker 给低分是正确的，应正常参与评分过滤，避免幽灵引用污染 context
    """
    kept = []
    for d, lbl in classified:
        if lbl == GradeLabel.INCORRECT and _is_about_page(d) and is_identity:
            lbl = GradeLabel.AMBIGUOUS  # 豁免：仅身份查询
        if lbl != GradeLabel.INCORRECT:
            kept.append(d)
    return kept[:top_k]


# ----------------------------------------------------------------------------
# Phase D（混合）：AMBIGUOUS 模糊带 LLM 裁决
#   模糊带仅 ~4% 文档（ct=0.3/it=0.2 校准结果），调一次便宜 LLM（GRADE_MODEL, temp=0）
#   做二分类（relevant/irrelevant），把模糊文档推到确定侧。成本接近 Phase A、质量接近
#   CRAG 原意。失败/未覆盖的文档保守保留（不影响召回）。
# ----------------------------------------------------------------------------
_JUDGE_PROMPT = """你是一个文档相关性评审员。给定用户问题和若干文档片段，请判断每个片段是否与问题相关（即是否包含可能帮助回答该问题的信息，部分相关也算相关）。

用户问题：
{query}

文档：
{docs}

请只输出一个 JSON 对象，键为文档编号（字符串），值为 "relevant" 或 "irrelevant"。例如：{{"1":"relevant","2":"irrelevant"}}。
不要输出任何额外文字。"""


def _parse_judge(text: str, n: int) -> Dict[int, str]:
    """把 LLM 输出解析成 {0-based 局部索引: "correct"|"incorrect"}。

    先试严格 JSON，失败再正则兜底。索引越界/解析失败的文档会被忽略（保守保留）。
    """
    text = text or ""
    out: Dict[int, str] = {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    ki = int(k)
                except (TypeError, ValueError):
                    continue
                if ki < 1 or ki > n:
                    continue
                vs = str(v).strip().lower()
                if vs in ("relevant", "correct", "相关", "1", "true"):
                    out[ki - 1] = "correct"
                elif vs in ("irrelevant", "incorrect", "无关", "0", "false"):
                    out[ki - 1] = "incorrect"
            return out
    except Exception:
        pass
    for m in re.finditer(
        r"(\d+)\s*[:\-]\s*[\"']?(relevant|irrelevant|correct|incorrect|相关|无关)",
        text,
        re.I,
    ):
        ki = int(m.group(1))
        if 1 <= ki <= n:
            out[ki - 1] = "correct" if m.group(2).lower() in ("relevant", "correct", "相关") else "incorrect"
    return out


def judge_ambiguous(query: str, docs: List[DocumentChunk]) -> Dict[int, GradeLabel]:
    """Phase D：对 AMBIGUOUS 带文档批量调 LLM 裁决，返回 {局部索引: GradeLabel}。

    返回空 dict = 未裁决（保守：这些文档保留进 context）。仅在确有模糊文档时调用，
    故单次查询通常只裁决 0~2 篇，成本低。
    """
    if not docs:
        return {}
    snippets = [f"[{i + 1}] (标题){d.title}\n{(d.content or '')[:600]}" for i, d in enumerate(docs)]
    prompt = _JUDGE_PROMPT.format(query=query, docs="\n\n".join(snippets))
    try:
        with timed_stage("crag_judge", count=len(docs)):
            resp = get_grade_llm().invoke(prompt).content
    except Exception:
        logger.exception("crag judge failed; keep ambiguous as-is")
        return {}
    verdicts = _parse_judge(resp, len(docs))
    return {
        idx: (GradeLabel.CORRECT if v == "correct" else GradeLabel.INCORRECT)
        for idx, v in verdicts.items()
    }


def _refine_with_judge(query: str, context: List[DocumentChunk], is_identity: bool = False) -> List[DocumentChunk]:
    """对最终 context 里仍落在 AMBIGUOUS 带的文档做 LLM 裁决，丢弃被判 irrelevant 的。

    context 可能来自 KB 精炼或 KB+web 合并 rerank，都带最新 score，可重新分类。
    只裁决模糊带，明确 CORRECT/INCORRECT 不进 LLM。
    About 页**仅身份查询时**豁免不送 judge（非身份查询中 about 应正常裁决）。
    """
    amb = [
        (i, d)
        for i, d in enumerate(context)
        if _label(d.score, CORRECT_THRESHOLD, INCORRECT_THRESHOLD) == GradeLabel.AMBIGUOUS
        and not (_is_about_page(d) and is_identity)  # P2: 仅身份查询豁免 about
    ]
    if not amb:
        return context
    amb_docs = [d for _, d in amb]
    verdicts = judge_ambiguous(query, amb_docs)
    to_drop = {
        orig_i
        for local_i, (orig_i, _) in enumerate(amb)
        if verdicts.get(local_i) == GradeLabel.INCORRECT
    }
    if not to_drop:
        return context
    return [d for i, d in enumerate(context) if i not in to_drop]


def _generate_answer(query: str, context: List[DocumentChunk], used_web: bool) -> SearchResponse:
    if not context:
        return SearchResponse(
            answer="No relevant content was found.", citations=[], fallback=True, mode="not_found"
        )
    citations = _dedupe_citations(context)
    n_sources = len(context)
    prompt = _GENERATE_PROMPT.format(
        n_sources=n_sources,
        context="\n\n".join(
            f"[{i + 1}] ({d.doc_type}) {d.title}\n{d.content}" for i, d in enumerate(context)
        ),
        query=query,
    )
    try:
        with timed_stage("crag_generate", count=n_sources):
            answer = get_gen_llm().invoke(prompt).content
    except Exception:
        logger.exception("crag generate failed")
        return SearchResponse(
            answer="Answer generation failed. Please try again later.",
            citations=citations, fallback=True, mode="error",
        )
    mode = "web" if used_web else "rag"
    return SearchResponse(
        answer=_clamp_citations(answer, n_sources), citations=citations, fallback=False, mode=mode
    )


def _prepare_context(query: str, top_k: int = 5) -> Tuple[List[DocumentChunk], bool]:
    """CRAG 核心编排：检索 -> 3 类评分 -> web 门控 -> 知识精炼 -> 返回 context。"""
    s = get_settings()
    identity = is_identity_query(query)  # P1+P2: 身份查询标识，控制 web 门控 + about 豁免
    candidate_k = max(top_k, s.RETRIEVAL_CANDIDATE_K)
    try:
        with timed_stage("crag_retrieve", query=query):
            docs = retrieve_with_rerank(query, top_k=candidate_k)
    except Exception:
        logger.exception("crag retrieve failed")
        return [], False

    classified = classify_documents(docs)
    need_web = decide_web_search(query, classified)

    used_web = False
    if need_web and s.WEB_SEARCH_ENABLED:
        try:
            with timed_stage("crag_web", query=query):
                web_docs = web_search(query)
        except Exception:
            logger.exception("crag web search failed")
            web_docs = []
        if web_docs:
            used_web = True
            # P3 修复：web 文档在合并 rerank 前给一个基础分（0.4），
            # 避免 KB 噪声文档（score 0.01~0.15）把有价值的 web 结果淹没。
            # 原问题：Case3 中 KB 全不相关(score<0.3)，web 的 CSDN 吴恩达课程
            # 也被排低 → LLM 忽略 → 答"无信息"。
            for wd in web_docs:
                if wd.score is None or wd.score < 0.35:
                    wd.score = 0.4
            combined = list(docs) + list(web_docs)
            context = rerank(query, combined, top_k)
        else:
            context = _keep_context(classified, top_k, is_identity=identity)
    else:
        context = _keep_context(classified, top_k, is_identity=identity)

    # Phase D：对最终 context 里仍落 AMBIGUOUS 带的文档调一次便宜 LLM 裁决，进一步精炼。
    if s.CRAG_PHASE_D and context:
        context = _refine_with_judge(query, context, is_identity=identity)

    return context, used_web


def run_rag_crag(query: str, top_k: int = 5) -> SearchResponse:
    context, used_web = _prepare_context(query, top_k)
    return _generate_answer(query, context, used_web)


def stream_rag_crag(query: str, top_k: int = 5) -> Iterator[tuple]:
    context, used_web = _prepare_context(query, top_k)
    citations = _dedupe_citations(context)
    mode = "web" if used_web else "rag"
    yield "sources", {"citations": [c.model_dump() for c in citations], "mode": mode}
    if not context:
        yield "done", SearchResponse(
            answer="No relevant content was found.", citations=citations, fallback=True, mode="not_found"
        ).model_dump()
        return
    n_sources = len(context)
    prompt = _GENERATE_PROMPT.format(
        n_sources=n_sources,
        context="\n\n".join(
            f"[{i + 1}] ({d.doc_type}) {d.title}\n{d.content}" for i, d in enumerate(context)
        ),
        query=query,
    )
    parts: list[str] = [_STREAM_PREFIX]
    yield "token", {"text": _STREAM_PREFIX}
    try:
        with timed_stage("crag_generate", count=n_sources):
            for chunk in get_gen_llm().stream(prompt):
                token = getattr(chunk, "content", "") or ""
                if token:
                    parts.append(token)
                    yield "token", {"text": token}
    except Exception:
        logger.exception("crag stream generation failed")
        yield "error", {"message": "Answer generation failed. Please try again later."}
        yield "done", SearchResponse(
            answer="Answer generation failed. Please try again later.",
            citations=citations, fallback=True, mode="error",
        ).model_dump()
        return
    full = "".join(parts)
    full = _clamp_citations(full, n_sources) if n_sources > 0 else full
    yield "done", SearchResponse(answer=full, citations=citations, fallback=False, mode=mode).model_dump()


_STREAM_PREFIX = "根据检索到的资料，"
