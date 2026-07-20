"""用于缓存相同查询结果的小型 TTL 内存缓存。

NOTE：在 serverless（Vercel）环境中，每个冷启动实例都拥有独立的内存，
因此此缓存为尽力而为模式（按实例生效）。如需共享缓存，请使用 Vercel KV 或 Redis，
并替换以下两个函数。
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

from core.config import get_settings

# 内存字典 tuple中float是时间戳，第二个是SearchResponse对象
_store: Dict[str, Tuple[float, object]] = {}

"""
先判断是否能直接返回，避免做贵的操作
"""
def cache_get(key: str):
    s = get_settings()
    item = _store.get(key)
    if item is None:
        return None
    ts, value = item
    if time.time() - ts > s.CACHE_TTL:
        _store.pop(key, None)
        return None
    return value


def cache_set(key: str, value: object) -> None:
    _store[key] = (time.time(), value)
