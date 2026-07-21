"""Pydantic request / response models for the RAG API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    history: list["ConversationTurn"] = Field(default_factory=list, max_length=2)


class ConversationTurn(BaseModel):
    role: Literal["user"] = "user"
    content: str = Field(..., min_length=1, max_length=2000)


class Citation(BaseModel):
    title: str
    url: str
    snippet: str = ""
    source: Literal["post", "page", "pdf", "web"] = "post"
    # rerank 相关性分数（0~1）。用于前端按置信度展示/排序，也作为
    # CITATION_MIN_SCORE 过滤的依据。未经过滤时为 rerank 原始分数。
    score: float = 0.0


class SearchResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    # 表示前端是否需要回退，当后端无法给出回答时表示需要前端兜底了，true表示后端无RAG答案
    fallback: bool = False  # True when we could not produce a RAG answer
    mode: str = "rag"  # "rag" | "web" | "not_found" | "error"
