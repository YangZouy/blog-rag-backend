"""BoCha web search fallback."""
from __future__ import annotations
import logging
import re
from typing import List
import requests
from core.auth import add_web_spend, web_budget_ok
from core.config import get_settings
from data.parse_hexo import DocumentChunk
logger = logging.getLogger("blog-rag")
_BOCHA_ENDPOINT = "https://api.bochaai.com/v1/web-search"
def _lightweight_filter(chunks: List[DocumentChunk], query: str) -> List[DocumentChunk]:
    terms = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", query.lower())
    if not terms: return chunks
    scored=[]
    for chunk in chunks:
        matches=sum(term in f"{chunk.title} {chunk.content}".lower() for term in terms)
        if matches: chunk.score=matches/len(terms); scored.append((matches,chunk))
    return [chunk for _,chunk in sorted(scored, key=lambda item:item[0], reverse=True)] if scored else chunks
def web_search(query: str, max_results: int | None = None) -> List[DocumentChunk]:
    settings=get_settings()
    if not settings.WEB_SEARCH_ENABLED or not settings.BOCHA_API_KEY or not web_budget_ok(): return []
    limit=max(1, max_results or getattr(settings, "WEB_SEARCH_MAX_PER_QUERY", 1))
    try:
        response=requests.post(_BOCHA_ENDPOINT, json={"query":query,"count":limit,"freshness":"noLimit"}, headers={"Authorization":f"Bearer {settings.BOCHA_API_KEY}"}, timeout=10)
        response.raise_for_status(); data=response.json()
    except Exception:
        logger.exception("web search request failed"); return []
    results=data.get("data",{}).get("webPages",{}).get("value") or data.get("webPages",{}).get("value") or []
    chunks=[]
    for index,item in enumerate(results[:limit]):
        url=item.get("url") or item.get("link") or ""
        if url: chunks.append(DocumentChunk(slug=f"web-{index}", title=item.get("name") or item.get("title") or "Web result", url=url, content=item.get("snippet") or item.get("description") or "", doc_type="web", chunk_index=0))
    filtered=_lightweight_filter(chunks,query); add_web_spend(0.01*len(filtered)); return filtered
