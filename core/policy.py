"""Deterministic policy checks that must run before any model or retrieval call."""
from __future__ import annotations

import re
from enum import Enum


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    PROMPT_INJECTION = "prompt_injection"
    AUTHORIZATION_DENIED = "authorization_denied"


_PROMPT_INJECTION = re.compile(
    r"(?:忽略|无视|绕过).{0,20}(?:系统|提示词|指令|检索|文章|资料|规则)|"
    r"(?:输出|泄露|展示|告诉我).{0,20}(?:系统提示词|内部提示词|内部评测|评测答案|隐藏指令)",
    re.IGNORECASE,
)
_AUTHORIZATION_BOUNDARY = re.compile(
    r"(?:导出|下载|查看|读取|提供).{0,24}(?:其他用户|别的用户|所有用户).{0,24}(?:对话|记录|trace|轨迹|引用|数据)|"
    r"(?:其他用户|别的用户|所有用户).{0,24}(?:对话|记录|trace|轨迹|引用|数据).{0,24}(?:导出|下载|查看|读取|提供)|"
    r"(?:读取|查看|输出|发送|发给我|提供).{0,24}(?:\.env|api\s*key|密钥|访问令牌|token)",
    re.IGNORECASE,
)


def evaluate_policy(query: str) -> PolicyDecision:
    """Return a narrow, auditable decision for known system-boundary requests."""
    text = (query or "").strip()
    if _PROMPT_INJECTION.search(text):
        return PolicyDecision.PROMPT_INJECTION
    if _AUTHORIZATION_BOUNDARY.search(text):
        return PolicyDecision.AUTHORIZATION_DENIED
    return PolicyDecision.ALLOW
