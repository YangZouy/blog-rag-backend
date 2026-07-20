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
    # 先拉大候选池再交给 LLM 筛选，避免只取 top5 漏掉长尾相关片段
    RETRIEVAL_CANDIDATE_K: int = 30
    GRADE_CANDIDATE_K: int = 12
    GRADE_MAX_CONCURRENCY: int = 4
    GRADE_SKIP_SCORE: float = 0.50
    # 低分拒答门槛：即便 grader 误留了低分噪声块，也先不交生成模型，
    # 改走改写 / 联网。注意 embedding-3 的真实相关分在 0.52–0.65 区间，
    # 故门槛不能高于 0.50，否则会误杀本来就相关的片段。
    SCORE_RELEVANCE_THRESHOLD: float = 0.50

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
    # Cheap model for grade / query-rewrite (default: 智谱 glm-4-flash)
    # Falls back to ZHIPU_API_KEY when GRADE_API_KEY is empty.
    # ------------------------------------------------------------------
    GRADE_MODEL: str = "glm-4-flash"
    GRADE_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4/"
    GRADE_API_KEY: str = ""

    # ------------------------------------------------------------------
    # Web search (default ON, 博查 BoCha)
    # ------------------------------------------------------------------
    WEB_SEARCH_ENABLED: bool = True
    BOCHA_API_KEY: str = ""
    WEB_SEARCH_MAX_PER_QUERY: int = 1
    DAILY_WEB_BUDGET: float = 1.0  # CNY; over budget -> disable web for the day

    # ------------------------------------------------------------------
    # CRAG（Corrective RAG）开关
    # ------------------------------------------------------------------
    # 灰度总开关：False 时 serve.py 仍走旧 rag_graph（run_rag/stream_rag）。
    # 设为 true 后切到 CRAG 旁路（api.crag）。默认关，待校准 + 端到端验证后再开。
    CRAG_ENABLED: bool = False
    # Phase D 混合 judge：对 AMBIGUOUS 模糊带文档调一次便宜 LLM（GRADE_MODEL）裁决。
    # 仅在 CRAG 主路激活时生效。默认开，但模糊带仅 ~4% 文档，成本极低。
    CRAG_PHASE_D: bool = True

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

    """
    @property让一个方法在使用时看起来像普通属性一样，不用写括号
    """
    @property
    def grade_api_key(self) -> str:
        return self.GRADE_API_KEY or self.ZHIPU_API_KEY

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
