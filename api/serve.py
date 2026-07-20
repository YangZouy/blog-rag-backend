"""FastAPI application exposing RAG search endpoints."""
from __future__ import annotations
import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from api.models import SearchRequest, SearchResponse
from api.rag_graph import run_rag, stream_rag
from core.auth import rate_limit, verify_api_key
from core.cache import cache_get, cache_set
from core.bm25 import warm_bm25
from core.config import get_settings
from core.qdrant_client import warm_qdrant
from core.rerank import warm_reranker
s = get_settings(); logger = logging.getLogger(__name__); logging.getLogger("blog-rag").setLevel(getattr(logging, s.LOG_LEVEL.upper(), logging.INFO))


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    warm_qdrant()
    if s.WARMUP_ON_START:
        logger.info("warming up bm25 index and reranker model...")
        warm_bm25()
        warm_reranker()
        logger.info("warmup done")
    yield


app = FastAPI(title="Blog RAG Search", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=s.allowed_origins_list, allow_methods=["POST", "GET", "OPTIONS"], allow_headers=["*"])
@app.get("/health")
def health() -> dict: return {"status": "ok"}
def _cache_key(req: SearchRequest) -> str:
    return f"rag\x1f{req.query}\x1f{req.top_k}"
def _sse(event: str, data: dict) -> str: return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key), Depends(rate_limit)])
def search(req: SearchRequest, request: Request) -> SearchResponse:
    key = _cache_key(req); cached = cache_get(key)
    if cached is not None: return cached
    try: response = run_rag(req.query, top_k=req.top_k)
    except Exception:
        logger.exception("RAG search failed for query=%r", req.query)
        return SearchResponse(answer="Search failed. Please try again later.", fallback=True, mode="error")
    cache_set(key, response); return response
@app.post("/search/stream", dependencies=[Depends(verify_api_key), Depends(rate_limit)])
def search_stream(req: SearchRequest, request: Request) -> StreamingResponse:
    key = _cache_key(req)
    def events() -> Iterator[str]:
        cached = cache_get(key)
        if isinstance(cached, SearchResponse):
            yield _sse("sources", {"citations": [citation.model_dump() for citation in cached.citations], "mode": cached.mode})
            if cached.answer: yield _sse("token", {"text": cached.answer})
            yield _sse("done", cached.model_dump()); return
        for event, data in stream_rag(req.query, req.top_k):
            yield _sse(event, data)
            if event == "done": cache_set(key, SearchResponse.model_validate(data))
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
