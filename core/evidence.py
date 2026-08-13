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

    @property
    def can_answer(self) -> bool:
        return self.status is not EvidenceStatus.INSUFFICIENT


def _has_reliable_hit(chunks: Sequence[DocumentChunk], threshold: float) -> bool:
    return bool(chunks) and (chunks[0].score or 0.0) >= threshold


def assess_evidence(
    plan: RetrievalPlan,
    chunks: Sequence[DocumentChunk],
    chunks_by_query: Mapping[str, Sequence[DocumentChunk]],
    threshold: float,
) -> EvidenceAssessment:
    """Assess score confidence and coverage of a bounded retrieval plan."""
    if not _has_reliable_hit(chunks, threshold):
        return EvidenceAssessment(EvidenceStatus.INSUFFICIENT)
    if not plan.is_complex:
        return EvidenceAssessment(EvidenceStatus.SUFFICIENT)

    missing = tuple(
        sub_query for sub_query in plan.queries
        if not _has_reliable_hit(chunks_by_query.get(sub_query, ()), threshold)
    )
    if not missing:
        return EvidenceAssessment(EvidenceStatus.SUFFICIENT)
    if len(missing) == len(plan.queries):
        return EvidenceAssessment(EvidenceStatus.INSUFFICIENT, missing)
    return EvidenceAssessment(EvidenceStatus.PARTIAL, missing)
