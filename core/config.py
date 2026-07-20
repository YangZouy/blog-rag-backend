"""博客 RAG 后端的集中配置。

所有密钥 / 模型名称 / 基础 URL 均通过环境变量读取（基于 pydantic-settings）。
任何敏感信息都不会硬编码在代码中。
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ------------------------------------------------------------------
    # Qdrant vector store
    # ------------------------------------------------------------------
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "blog_chunks"

    # ------------------------------------------------------------------
    # Retrieval tuning
    # ------------------------------------------------------------------
    # 召回后交给 reranker 的候选数。eval 已验证 top5 可覆盖答案，8 保留余量。
    RETRIEVAL_CANDIDATE_K: int = 8
    # 最终送入生成模型的上下文块数；同一文章最多保留一个最高分 chunk。
    GENERATION_CONTEXT_K: int = 3
    # reranker 最高分低于此值时，视为没有足够相关的站内资料。
    RERANK_RELEVANCE_THRESHOLD: float = 0.30

    # ------------------------------------------------------------------
    # Embedding model (default: 智谱 embedding-3, 2048-dim)
    # 8192 tokens 的输入 编码格式：float或base64
    # API限制：每次最多64条文本
    # ------------------------------------------------------------------
    EMBED_MODEL: str = "embedding-3"
    EMBED_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4/"
    EMBED_DIM: int = 2048
    EMBED_BATCH_SIZE: int = 64
    ZHIPU_API_KEY: str = ""
    QDRANT_UPSERT_BATCH_SIZE: int = 32
    QDRANT_WRITE_TIMEOUT: int = 120
    # 单次 Qdrant 请求超时（秒）。云端 TLS 冷连接握手可能 >3s，过短会 SSL 握手超时；
    # 30s 兼顾冷启动与正常请求，warm 后实际请求通常 <1s。
    QDRANT_READ_TIMEOUT: int = 30
    QDRANT_WARMUP_ENABLED: bool = True
    # 服务启动时预加载 BM25 索引与 reranker 模型，避免首次请求冷启动
    WARMUP_ON_START: bool = True

    # ------------------------------------------------------------------
    # Generation model (default: DeepSeek-chat)
    # ------------------------------------------------------------------
    GEN_MODEL: str = "deepseek-chat"
    GEN_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_API_KEY: str = ""

    # ------------------------------------------------------------------
    # API server 加固
    # ------------------------------------------------------------------
    API_KEY: str = ""  # if non-empty, clients must send x-api-key / ?api_key=
    ALLOWED_ORIGINS: str = "*"  # comma separated; set to your GitHub Pages domain
    RATE_LIMIT_PER_MIN: int = 10
    # 检索后答案保存时间 600s就是10min
    CACHE_TTL: int = 600

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    SITE_URL: str = ""  # e.g. https://yangzouy.github.io  (used to build URLs)

    @property
    def allowed_origins_list(self) -> list[str]:
        if self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

"""
配置对象不用每次都重新读.env，该函数返回结果记住一次，以后复用
"""
@lru_cache
def get_settings() -> Settings:
    return Settings()
