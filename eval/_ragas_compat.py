# -*- coding: utf-8 -*-
"""ragas 0.4.x 兼容垫片。

ragas 0.4.x 在 `ragas/llms/base.py` 顶部无条件
`from langchain_community.chat_models.vertexai import ChatVertexAI` 且
`from langchain_community.llms import VertexAI`，但这两个符号在
langchain-community>=0.3 中已被移除。本项目运行时是 langchain 1.x 栈
（langchain-community 0.4.2），因此直接 `import ragas` 会报
`ModuleNotFoundError: langchain_community.chat_models.vertexai`。

ragas 只在 `MULTIPLE_COMPLETION_SUPPORTED` 列表里对这两个类做 `isinstance`
判断，从不实例化，所以注入两个无害的 stub 类即可让 ragas 正常 import。
本垫片对生产零影响：ragas 仅出现在 requirements-dev.txt，生产部署不安装。

在 eval_ragas.py 顶部 import 本模块即可（import 时自动执行 _install）。
"""
from __future__ import annotations

import sys
import types


class _ChatVertexAI:  # stub，仅用于 isinstance 判断
    pass


class _VertexAI:  # stub，仅用于 isinstance 判断
    pass


def _install() -> None:
    vertexai_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    vertexai_mod.ChatVertexAI = _ChatVertexAI
    sys.modules.setdefault("langchain_community.chat_models.vertexai", vertexai_mod)

    import langchain_community.llms as llms_mod

    if not hasattr(llms_mod, "VertexAI"):
        llms_mod.VertexAI = _VertexAI


_install()
