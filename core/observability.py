"""Small logging helpers shared by the RAG request path."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("blog-rag")

@contextmanager
def timed_stage(stage: str, **fields: object) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "rag_stage stage=%s duration_ms=%.1f%s",
            stage,
            duration_ms,
            "".join(f" {key}={value!r}" for key, value in fields.items()),
        )
