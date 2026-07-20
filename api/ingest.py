"""将 Hexo 源仓库导入 Qdrant 向量存储。

本地运行：
    python -m api.ingest --repo /path/to/hexo-source

CI 中（参见 scripts/ingest_ci.sh）：先检出源仓库，然后调用此脚本。
写入操作是幂等的（点 ID = slug:chunk_index），因此重复运行是安全的。
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid

from qdrant_client.models import PointStruct

from core.config import get_settings
from core.embeddings import get_embeddings
from core.qdrant_client import ensure_collection, get_qdrant
from data.parse_hexo import parse_hexo_repo
from data.parse_pdf import parse_pdfs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")


def run_ingest(repo_path: str, recreate: bool = False) -> int:
    s = get_settings()
    ensure_collection(recreate=recreate)
    client = get_qdrant()
    embed = get_embeddings()

    chunks = parse_hexo_repo(repo_path) + parse_pdfs(repo_path)
    docs = [c for c in chunks if c.content]
    if not docs:
        logger.warning("no ingestable documents found in %s", repo_path)
        return 0

    logger.info(
        "embedding %d chunks in batches of %d...",
        len(docs),
        s.EMBED_BATCH_SIZE,
    )
    # 嵌入模型API输入最大为64
    # 用组合文本（标题/标签/章节/正文）做 embedding，而非仅正文
    vectors = embed.embed_documents(
        [c.embed_text() for c in docs],
        chunk_size=s.EMBED_BATCH_SIZE,
    )

    points = [
        # 定义向量点的数据结构
        PointStruct(
            # Qdrant 要求使用 UUID 或无符号整数作为 ID；此处根据
            # slug:chunk_index 生成一个稳定的 ID，以保证写入操作的幂等性。
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c.slug}:{c.chunk_index}")),
            vector=v,
            payload={"chunk": c.to_payload(), "doc_type": c.doc_type, "tags": c.tags, "slug": c.slug},
        )
        for c, v in zip(docs, vectors)
    ]

    # 分批写入向量数据库
    # upsert容易超时，现在分批写入，默认值为32
    batch = s.QDRANT_UPSERT_BATCH_SIZE
    for i in range(0, len(points), batch):
        client.upsert(
            collection_name=s.QDRANT_COLLECTION,
            points=points[i : i + batch],
            timeout=s.QDRANT_WRITE_TIMEOUT,
        )
    logger.info("ingested %d chunks into '%s'", len(points), s.QDRANT_COLLECTION)
    return len(points)


def main() -> None:
    # 创建一个参数解析器
    parser = argparse.ArgumentParser(description="Ingest a Hexo repo into Qdrant")
    parser.add_argument("--repo", required=True, help="path to the Hexo repo root")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and recreate the collection before ingest (use after URL/schema changes)",
    )
    args = parser.parse_args()
    # 取出用户传进来的值
    n = run_ingest(args.repo, recreate=args.recreate)
    print(f"ingested {n} chunks")
    sys.exit(0 if n >= 0 else 1)


if __name__ == "__main__":
    main()
