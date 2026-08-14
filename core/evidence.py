"""
证据评估器
Deterministic evidence gates for the controlled RAG pipeline.

This module turns retrieval scores and sub-query coverage into one of three
answer boundaries. It does not generate text or invoke a model.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence
import re

from core.planning import RetrievalPlan
from data.parse_hexo import DocumentChunk


class EvidenceStatus(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class EvidenceAssessment:
    status: EvidenceStatus
    missing_aspects: tuple[str, ...] = ()
    aspect_details: tuple["AspectEvidence", ...] = ()

    @property
    def can_answer(self) -> bool:
        return self.status is not EvidenceStatus.INSUFFICIENT


@dataclass(frozen=True)
class AspectEvidence:
    query: str
    top_score: float
    threshold: float
    supported: bool
    reason: str


def _has_reliable_hit(chunks: Sequence[DocumentChunk], threshold: float) -> bool:
    return bool(chunks) and (chunks[0].score or 0.0) >= threshold


_QUANTIFIED_CLAIM = re.compile(
    r"(?:多少|qps|sla|故障率|成功率|收入|成本|付费.{0,4}客户|独立访客|客户数)",
    re.IGNORECASE,
)
_NUMERIC_EVIDENCE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:%|％|qps|元|万元|万|家|个|次|秒|ms|毫秒|人)|"
    r"(?:qps|sla|故障率|成功率|收入|成本|客户|访客).{0,24}\d+)",
    re.IGNORECASE,
)
_QUANTIFIED_ANCHORS = ("qps", "sla", "故障率", "成功率", "收入", "成本", "访客", "客户")


def _has_direct_quantified_support(query: str, evidence: str) -> bool:
    anchors = [anchor for anchor in _QUANTIFIED_ANCHORS if anchor.lower() in query.lower()]
    if not anchors:
        return bool(_NUMERIC_EVIDENCE.search(evidence))
    for anchor in anchors:
        escaped = re.escape(anchor)
        near_number = re.search(
            rf"(?:{escaped}.{{0,32}}\d|\d.{{0,32}}{escaped})",
            evidence,
            re.IGNORECASE | re.DOTALL,
        )
        if not near_number:
            return False
    return True


def _assess_aspect(
    query: str,
    chunks: Sequence[DocumentChunk],
    threshold: float,
    quantified_threshold: float,
) -> AspectEvidence:
    is_quantified = bool(_QUANTIFIED_CLAIM.search(query))
    effective_threshold = quantified_threshold if is_quantified else threshold
    top_score = max(((chunk.score or 0.0) for chunk in chunks), default=0.0)
    reliable = [chunk for chunk in chunks if (chunk.score or 0.0) >= effective_threshold]
    if not reliable:
        return AspectEvidence(
            query, round(top_score, 4), effective_threshold, False, "low_score"
        )
    if not is_quantified:
        return AspectEvidence(
            query, round(top_score, 4), effective_threshold, True, "supported"
        )
    evidence = "\n".join(f"{chunk.title}\n{chunk.content}" for chunk in reliable)
    supported = _has_direct_quantified_support(query, evidence)
    return AspectEvidence(
        query,
        round(top_score, 4),
        effective_threshold,
        supported,
        "supported" if supported else "missing_quantified_evidence",
    )


def _has_supported_hit(
    query: str,
    chunks: Sequence[DocumentChunk],
    threshold: float,
    quantified_threshold: float | None = None,
) -> bool:
    return _assess_aspect(
        query, chunks, threshold, quantified_threshold or threshold
    ).supported


def assess_evidence(
    plan: RetrievalPlan,
    chunks: Sequence[DocumentChunk],
    chunks_by_query: Mapping[str, Sequence[DocumentChunk]],
    threshold: float,
    quantified_threshold: float | None = None,
) -> EvidenceAssessment:
    """Assess score confidence and coverage of a bounded retrieval plan."""
    quantified_threshold = quantified_threshold or threshold
    if not plan.is_complex:
        detail = _assess_aspect(
            plan.original_query, chunks, threshold, quantified_threshold
        )
        if detail.supported:
            status = EvidenceStatus.SUFFICIENT
        elif detail.reason == "missing_quantified_evidence":
            # 相关文档已检索到（top 分达标），仅缺具体数字 → 可部分作答，
            # 不应整句拒答。只有真正无可靠文档（low_score）才判 INSUFFICIENT。
            status = EvidenceStatus.PARTIAL
        else:
            status = EvidenceStatus.INSUFFICIENT
        return EvidenceAssessment(
            status,
            () if detail.supported else (plan.original_query,),
            (detail,),
        )

    details = tuple(
        _assess_aspect(
            sub_query,
            chunks_by_query.get(sub_query, ()),
            threshold,
            quantified_threshold,
        )
        for sub_query in plan.queries
    )
    missing = tuple(detail.query for detail in details if not detail.supported)
    if not missing:
        return EvidenceAssessment(EvidenceStatus.SUFFICIENT, (), details)
    if len(missing) == len(details):
        return EvidenceAssessment(EvidenceStatus.INSUFFICIENT, missing, details)
    return EvidenceAssessment(EvidenceStatus.PARTIAL, missing, details)
