"""本地 Faiss 向量存储层（取代 Qdrant，实现自托管、零网络检索）。

设计要点：
- 向量检索在服务器进程内完成，零跨境网络；量级 ~2000×2048 时单次查询 1~10ms。
- 使用 faiss.IndexFlatIP（内积）+ 向量 L2 归一化，等价于 Qdrant 的 COSINE 距离。
- 元数据（chunk 字段）与 faiss 行一一对应，持久化为 JSON；faiss 索引持久化为二进制文件。
- 支持 upsert（按 (slug, chunk_index) 幂等合并）、delete_by_slug（增量入库清理）、
  search（含 doc_type/tags 过滤）、get_all（BM25 预热用）、reset（--recreate 清空）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:  # 运行时才安装；启动时 get_vector_store 会报清晰错误
    faiss = None

from core.config import get_settings
from data.parse_hexo import DocumentChunk

logger = logging.getLogger("blog-rag")

ChunkScore = Tuple[DocumentChunk, float]
Key = Tuple[str, int]


class LocalVectorStore:
    def __init__(self) -> None:
        s = get_settings()
        self.dim = s.EMBED_DIM
        self.index_path = s.FAISS_INDEX_PATH
        self.meta_path = s.FAISS_META_PATH
        self._lock = threading.Lock()
        self.index = None
        self.metadata: List[dict] = []
        self._vecs: np.ndarray = np.empty((0, self.dim), dtype="float32")
        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if faiss is None:
            raise RuntimeError("faiss 未安装，请先 `pip install faiss-cpu`")
        if os.path.isfile(self.index_path) and os.path.isfile(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, encoding="utf-8") as f:
                    self.metadata = json.load(f)
                n = self.index.ntotal
                if n > 0:
                    self._vecs = np.vstack(
                        [self.index.reconstruct(i) for i in range(n)]
                    ).astype("float32")
                else:
                    self._vecs = np.empty((0, self.dim), dtype="float32")
                logger.info("loaded faiss index: %d vectors", n)
                return
            except Exception:
                logger.warning("加载 faiss 索引失败，将以空索引启动", exc_info=True)
        self._init_empty()

    def _init_empty(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        self._vecs = np.empty((0, self.dim), dtype="float32")

    def _save(self) -> None:
        d = os.path.dirname(self.index_path)
        if d:
            os.makedirs(d, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def _rebuild(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        if self._vecs.shape[0] > 0:
            self.index.add(self._vecs)
        self._save()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(v) -> np.ndarray:
        arr = np.asarray(v, dtype="float32").ravel()
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._init_empty()
            self._save()

    def upsert(self, items: List[Tuple[DocumentChunk, List[float]]]) -> int:
        """幂等写入：按 (slug, chunk_index) 合并，已存在则覆盖，否则追加。"""
        with self._lock:
            key_to_idx: Dict[Key, int] = {
                (m.get("slug"), m.get("chunk_index")): i
                for i, m in enumerate(self.metadata)
            }
            updates: Dict[int, Tuple[np.ndarray, dict]] = {}
            new_vecs: List[np.ndarray] = []
            new_meta: List[dict] = []
            for chunk, vector in items:
                payload = chunk.to_payload()
                norm = self._normalize(vector)
                key: Key = (chunk.slug, chunk.chunk_index)
                if key in key_to_idx:
                    updates[key_to_idx[key]] = (norm, payload)
                else:
                    new_vecs.append(norm)
                    new_meta.append(payload)
            for idx, (norm, payload) in updates.items():
                self._vecs[idx] = norm
                self.metadata[idx] = payload
            if new_vecs:
                add = np.vstack(new_vecs)
                self._vecs = np.vstack([self._vecs, add]) if self._vecs.shape[0] else add
                self.metadata.extend(new_meta)
            self._rebuild()
            return len(items)

    def delete_by_slug(self, slugs) -> None:
        slugs = set(slugs or [])
        if not slugs:
            return
        with self._lock:
            keep = [
                (i, m) for i, m in enumerate(self.metadata) if m.get("slug") not in slugs
            ]
            if not keep:
                self.metadata = []
                self._vecs = np.empty((0, self.dim), dtype="float32")
            else:
                idxs = [i for i, _ in keep]
                self.metadata = [m for _, m in keep]
                self._vecs = np.vstack([self._vecs[i] for i in idxs]).astype("float32")
            self._rebuild()

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------
    def search(
        self,
        vector,
        top_k: int,
        doc_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ChunkScore]:
        """返回 top_k 个 (DocumentChunk, score)，可选按 doc_type/tags 过滤。

        faiss 不支持服务端结构化过滤，故先取较宽候选（top_k*4，至少 24），
        再在 Python 端过滤并截断，效果等价于 Qdrant 的 query_filter。
        """
        with self._lock:
            if self.index.ntotal == 0:
                return []
            q = self._normalize(vector).reshape(1, -1)
            k = min(max(top_k * 4, 24), self.index.ntotal)
            scores, idxs = self.index.search(q, k)
            scores = scores[0]
            idxs = idxs[0]
        results: List[ChunkScore] = []
        for score, i in zip(scores, idxs):
            if i < 0:
                continue
            chunk = DocumentChunk.from_payload(self.metadata[i])
            if doc_type and chunk.doc_type != doc_type:
                continue
            if tags and not set(tags).intersection(chunk.tags or []):
                continue
            chunk.score = float(score)
            results.append((chunk, float(score)))
        return results[:top_k]

    def get_all(self) -> List[DocumentChunk]:
        with self._lock:
            return [DocumentChunk.from_payload(m) for m in self.metadata]

    def count(self) -> int:
        return self.index.ntotal


@lru_cache(maxsize=1)
def get_vector_store() -> LocalVectorStore:
    return LocalVectorStore()


def warm_vector_store() -> None:
    """触发索引加载，确保服务启动即就绪（首次请求不再因加载而变慢）。"""
    try:
        get_vector_store()
        logger.info("faiss vector store warmed")
    except Exception:
        logger.warning(
            "faiss warmup failed; first request will build on demand", exc_info=True
        )
