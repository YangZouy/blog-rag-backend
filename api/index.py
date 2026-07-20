"""Vercel Serverless 入口。

Vercel 的 Python 运行时会自动检测 `api/index.py` 中的 ASGI `app`，
并将配置的路由映射到它（参见 vercel.json）。此文件应保持精简——
业务逻辑位于 api.serve 中。
"""
from api.serve import app

__all__ = ["app"]
