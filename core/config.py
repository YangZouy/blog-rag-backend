"""博客 RAG 后端的集中配置。

所有密钥 / 模型名称 / 基础 URL 均通过环境变量读取（基于 pydantic-settings）。
任何敏感信息都不会硬编码在代码中。
"""
from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ------------------------------------------------------------------
    # Retrieval tuning
    # ------------------------------------------------------------------
    # 召回后交给 reranker 的候选数。eval 已验证 top5 可覆盖答案，8 保留余量。
    RETRIEVAL_CANDIDATE_K: int = 8
    # The only retry action currently enabled: one evidence-insufficient query
    # may expand its candidate pool once, then the pipeline must terminate.
    RETRIEVAL_REMEDY_CANDIDATE_K: int = 16
    MAX_RETRIEVAL_ROUNDS: int = 2
    # 最终送入生成模型的上下文块数
    GENERATION_CONTEXT_K: int = 5
    # reranker 最高分低于此值时，视为没有足够相关的站内资料。
    RERANK_RELEVANCE_THRESHOLD: float = 0.30
    # Evidence gate uses the same calibrated reranker scale. A complex question
    # is only fully answerable when each planned sub-query clears this threshold.
    EVIDENCE_RELEVANCE_THRESHOLD: float = 0.30
    # 推荐阅读（citations）的置信度门槛，作用于 _dedupe_citations 的 retrieved 全集。
    # 生成上下文门槛（0.30）允许边缘相关；推荐阅读门槛更严，宁缺毋滥。
    CITATION_MIN_SCORE: float = 0.50

    # ------------------------------------------------------------------
    # Embedding model (智谱 embedding-3, 2048-dim，远程 API)
    # 8192 tokens 的输入 编码格式：float或base64
    # API限制：每次最多64条文本
    # ------------------------------------------------------------------
    EMBED_MODEL: str = "embedding-3"
    EMBED_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4/"
    EMBED_DIM: int = 2048
    EMBED_BATCH_SIZE: int = 64
    ZHIPU_API_KEY: str = ""
    # 服务启动时预加载 BM25 索引与 reranker 模型，避免首次请求冷启动
    WARMUP_ON_START: bool = True

    # ------------------------------------------------------------------
    # Reranker：local（本地ONNX int8，默认，无网络依赖）
    # ------------------------------------------------------------------
    RERANK_BACKEND: str = "local"
    # 本地 ONNX 模型仓库与文件（Xenova 转换的 bge-reranker-base int8 量化版，~280MB）
    RERANK_MODEL_REPO: str = "Xenova/bge-reranker-base"
    RERANK_ONNX_FILE: str = "onnx/model_quantized.onnx"
    # query+doc 拼接后的最大 token 长度。
    RERANK_MAX_LENGTH: int = 256
    # 进入 rerank 的候选池大小。
    # 从 20 降到 10：rerank 耗时与候选数近似线性，砍半约省一半 rerank 时间
    # （弱机可省 ~6s）；eval 显示 hybrid recall 已 1.000，rerank 仅提升排序(MRR 0.803→0.871)，
    # 小库里 top-8 结果几乎不受影响。
    RERANK_CANDIDATE_K: int = 10

    # 意图识别模型
    INTENT_LLM_MODEL: str = "glm-4-flash"
    ZHIPU_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4/"
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
    # Admin / 增量入库
    # ------------------------------------------------------------------
    # 独立令牌，用于 /admin/reload，CI调用时使用
    ADMIN_TOKEN: str = ""
    # 增量入库用的 slug→hash 状态文件（运行时生成，应 gitignore）
    # data/ingest_state.json：保存slug→content_hash
    # 增量入库需要知道相对上次 哪些变化了
    INGEST_STATE_PATH: str = os.path.join(ROOT, "data", "ingest_state.json")

    # ------------------------------------------------------------------
    # 本地 Faiss 向量存储（自托管，零网络检索）
    # ------------------------------------------------------------------
    # faiss 索引二进制（向量）+ 元数据 JSON（chunk 字段），运行时生成应 gitignore
    FAISS_INDEX_PATH: str = os.path.join(ROOT, "data", "vector_store.faiss")
    FAISS_META_PATH: str = os.path.join(ROOT, "data", "vector_store_meta.json")

    # ------------------------------------------------------------------
    # 本地博客仓库（服务器侧克隆副本，入库数据源）
    # 服务器 git clone 博客 GitHub 仓库到此路径；/admin/reload 会先 git pull 再增量入库。
    # 为空时 /admin/reload 必须显式带 repo。本地开发可指向 D:/Blog。
    # ------------------------------------------------------------------
    BLOG_REPO_PATH: str = ""

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
