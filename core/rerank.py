"""
Cross-encoder reranker
默认后端：本地 ONNX int8 量化模型（bge-reranker-base），CPU 推理，无网络依赖。
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

import httpx

from core.config import get_settings
from data.parse_hexo import DocumentChunk

logger = logging.getLogger("blog-rag")
RERANK_API_TIMEOUT = 15

# 国内网络默认走 HF 镜像；仅当未显式设置（或为空/默认值）时启用镜像，
# 避免 setdefault 被环境中已有的 HF_ENDPOINT（可能指向不可达的 huggingface.co）覆盖。
_HF = os.environ.get("HF_ENDPOINT", "").rstrip("/")
if not _HF or _HF in ("https://huggingface.co", "http://huggingface.co"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------------------------------------------------------------------------
# Local ONNX backend
# ---------------------------------------------------------------------------

def _resolve_model_file(filename: str, models_dir: str) -> "str | None":
    """优先用本地已存在的模型文件（从服务器/镜像拷过来的），完全离线。

    先查直接相对路径，再递归查找，兼容 hf_hub_download(local_dir=...) 可能带
    repo_id 前缀的不同布局。文件不存在返回 None（交给联网下载兜底）。
    """
    direct = os.path.join(models_dir, filename)
    if os.path.isfile(direct):
        return direct
    if os.path.isdir(models_dir):
        for root, _, files in os.walk(models_dir):
            if filename in files:
                return os.path.join(root, filename)
    return None

_LOCAL_LOCK = threading.Lock()
_LOCAL_RERANKER: Optional["_OnnxReranker"] = None

class _OnnxReranker:
    def __init__(self, repo: str, onnx_file: str, max_length: int) -> None:
      import numpy as np
      import onnxruntime as ort
      from huggingface_hub import hf_hub_download
      from tokenizers import Tokenizer

      # 模型路径
      models_dir = os.path.join(
          os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
          "models", "reranker",
      )

      # 本地已有模型文件则直接复用（离线，免去联网下载）；否则再尝试从 Hugging Face 拉取
      model_path = _resolve_model_file(onnx_file, models_dir)
      tokenizer_path = _resolve_model_file("tokenizer.json", models_dir)
      if model_path is None or tokenizer_path is None:
          logger.warning(
              "本地未找到 reranker 模型（%s / tokenizer.json），尝试联网下载（需能访问 Hugging Face 镜像）",
              onnx_file,
          )
          model_path = hf_hub_download(repo_id=repo, filename=onnx_file, local_dir=models_dir)
          tokenizer_path = hf_hub_download(
              repo_id=repo, filename="tokenizer.json", local_dir=models_dir
          )

      with open(tokenizer_path, encoding="utf-8") as fh:
          self.tokenizer = Tokenizer.from_str(fh.read())
      self.tokenizer.enable_truncation(max_length=max_length)
      self.tokenizer.enable_padding()

      so = ort.SessionOptions()
      # 限制线程数，避免与 uvicorn worker 抢核
      so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
      self.session = ort.InferenceSession(
          model_path, sess_options=so, providers=["CPUExecutionProvider"]
      )
      self.input_names = {i.name for i in self.session.get_inputs()}

    def score(self, query: str, texts: List[str]) -> List[float]:
        import numpy as np

        encodings = self.tokenizer.encode_batch([(query, t) for t in texts])
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        logits = self.session.run(None, feed)[0].reshape(-1)
        # sigmoid：logit -> 0~1 相关性概率，与 Jina relevance_score 同量纲
        return (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).tolist()

def _get_local_reranker() -> _OnnxReranker:
    global _LOCAL_RERANKER
    if _LOCAL_RERANKER is None:
        with _LOCAL_LOCK:
            if _LOCAL_RERANKER is None:
                s = get_settings()
                logger.info(
                    "loading local ONNX reranker: %s (%s)",
                    s.RERANK_MODEL_REPO,
                    s.RERANK_ONNX_FILE,
                )
                _LOCAL_RERANKER = _OnnxReranker(
                    s.RERANK_MODEL_REPO, s.RERANK_ONNX_FILE, s.RERANK_MAX_LENGTH
                )
    return _LOCAL_RERANKER

def _rerank_local(query: str, chunks: List[DocumentChunk], limit: int) -> List[DocumentChunk]:
    rerank = _get_local_reranker()
    scores = rerank.score(query, [c.embed_text() for c in chunks])
    for chunk, s in zip(chunks, scores):
        chunk.score = float(s)
    return sorted(chunks, key=lambda c: c.score or 0.0, reverse=True)[:limit]

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def rerank(query: str, chunks: List[DocumentChunk], limit: int) -> List[DocumentChunk]:
    """Score hybrid candidates and return the top `limit` chunks."""
    if not chunks:
        return []
    try:
        return _rerank_local(query, chunks, limit)
    except Exception:
        logger.exception("local rerank failed; falling back to hybrid score order")
        # 回退时 hybrid RRF 分数（~0.03 量纲）远低于相关性阈值，
        # 下游会将其视为「无可信站内命中」——宁可少推，不给错误高分。
        return sorted(chunks, key=lambda c: c.score or 0.0, reverse=True)[:limit]
    
def warm_reranker() -> None:
    """
    启动时预加载：本地后端做一次真实推理（首次会触发模型下载）
    """
    try:
        reranker = _get_local_reranker()
        reranker.score("warmup", ["预热文本"])
        logger.info("local ONNX reranker warmed up")
    except Exception:
        logger.exception("local reranker warmup failed; requests will retry lazily")


