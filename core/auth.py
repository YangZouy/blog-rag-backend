"""API 加固：密钥校验 + 单IP速率限制 + 每日全局调用量保护。

这些是 /search 端点使用的 FastAPI 依赖项。它们是基于内存的简单实现，
适用于个人博客；如需多实例生产环境部署，请改用 Redis 支持的存储方案。
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import date

from fastapi import HTTPException, Request, status

from core.config import get_settings

# 全局内存字典 key是IP，value是时间戳
_rate: defaultdict[str, list[float]] = defaultdict(list)
# web-search spend tracker (CNY) keyed by UTC day
_web_spend: dict[str, float] = {}

"""
模型api_key鉴权
防止别人恶意调用接口，刷模型额度
"""
async def verify_api_key(request: Request) -> None:
    # 取配置API，如果没设置，说明不鉴权，直接放行
    s = get_settings()
    if not s.API_KEY:
        return
    # 否则从请求头或query中取值
    key = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if key != s.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key"
        )

"""
防止短时间恶意高频请求
"""
async def rate_limit(request: Request) -> None:
    s = get_settings()
    if s.RATE_LIMIT_PER_MIN <= 0:
        return
    ip = request.client.host if request.client else "anon"
    now = time.time()
    window = _rate[ip]
    # 超过60s的就信息就不看了
    _rate[ip] = [t for t in window if now - t < 60]
    # 统计近60s该IP用户的使用频率，超过返回429
    if len(_rate[ip]) >= s.RATE_LIMIT_PER_MIN:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limited"
        )
    _rate[ip].append(now)


def web_budget_ok() -> bool:
    """当今日的网络搜索花费超过每日预算时返回 False。"""
    s = get_settings()
    if not s.WEB_SEARCH_ENABLED:
        return False
    # 将每日预算设定为0或者负数，表示不限制预算，允许无限制地联网搜索
    if s.DAILY_WEB_BUDGET <= 0:
        return True
    today = str(date.today())
    return _web_spend.get(today, 0.0) < s.DAILY_WEB_BUDGET


def add_web_spend(amount: float) -> None:
    today = str(date.today())
    _web_spend[today] = _web_spend.get(today, 0.0) + amount
