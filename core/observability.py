"""Small logging helpers shared by the RAG request path."""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator

logger = logging.getLogger("blog-rag")

# 进程级「首次真实请求」标记，用于在日志中标 cold_start / warm_request。
_req_lock = threading.Lock()
_first_request_done: Dict[str, bool] = {}


def request_type(tag: str = "req") -> str:
    """进程启动后第一次真实（非缓存命中）请求标记为 cold_start，之后为 warm_request。

    用于区分「首问因加载模型/索引而偏慢」与「稳态请求」，便于对比延迟。
    """
    with _req_lock:
        if not _first_request_done.get(tag, False):
            _first_request_done[tag] = True
            return "cold_start"
        return "warm_request"


@contextmanager
def timed_stage(stage: str, **fields: object) -> Iterator[Dict[str, object]]:
    """记录一段耗时的上下文管理器（支持块结束后追加字段）。

    用法：
        with timed_stage("embed_query", query=q) as f:
            vec = ...
        f["cache_hit"] = True   # 追加字段，会进入同一行日志

    日志格式固定为 `rag_stage stage=<name> duration_ms=<ms> <key>=<value> ...`，
    与既有 `grep rag_stage` / `grep stage=xxx` 完全兼容（仅新增字段，不改前缀）。
    """
    started = time.perf_counter()
    extra: Dict[str, object] = dict(fields)
    try:
        yield extra
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        extra["duration_ms"] = round(duration_ms, 1)
        parts = " ".join(f"{k}={v!r}" for k, v in extra.items() if k != "duration_ms")
        logger.info(
            "rag_stage stage=%s duration_ms=%.1f%s",
            stage,
            duration_ms,
            (" " + parts) if parts else "",
        )
