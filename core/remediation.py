"""Bounded remediation policy for evidence-insufficient retrievals.

The policy deliberately exposes a small action whitelist. It cannot request
tools, create arbitrary queries, or schedule a third retrieval round.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RemedyAction(str, Enum):
    EXPAND_CANDIDATES = "expand_candidates"


@dataclass(frozen=True)
class RemedyDecision:
    action: RemedyAction
    candidate_k: int


def choose_remedy(current_candidate_k: int, expanded_candidate_k: int) -> RemedyDecision | None:
    """Choose the only permitted first-round remediation, if it adds coverage."""
    if expanded_candidate_k <= current_candidate_k:
        return None
    return RemedyDecision(RemedyAction.EXPAND_CANDIDATES, expanded_candidate_k)
