# Blog RAG Search Backend

为个人 Hexo 博客提供语义检索与 AI 问答能力的后端服务。前端为右下角悬浮 AI 聊天窗。

**架构链路**：`query → 规则快路(问候/实时零成本短路) → [LLM分类 ∥ 检索并行] → 向量+BM25(RRF融合) → 本地ONNX重排 → DeepSeek生成 → SSE流式输出`

- 后端：FastAPI（systemd 托管，Nginx 反代）
- 向量库：Qdrant Cloud（持久化，2048 维 COSINE）
- Embedding：智谱 embedding-3 / 生成：DeepSeek-chat / 意图分类：glm-4-flash
- 重排：bge-reranker-base ONNX int8 量化，CPU 本地推理，无网络依赖
- 检索：Dense + BM25 (jieba) → RRF(k=60) 融合 → cross-encoder 重排
- 前端：原生 JS，按 stage/sources/token/done 四类 SSE 事件增量渲染

---

## 1. 本地开发

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  / macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填入 Qdrant / 智谱 / DeepSeek 密钥
python -m uvicorn api.serve:app --reload --port 8000
```

健康检查与搜索：

```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"如何部署博客","top_k":5}'
```

### 流式搜索

`POST /search/stream` 通过 `text/event-stream` 推送事件，顺序为：

| 事件 | 含义 |
|------|------|
| `stage` | 当前阶段（routing / retrieving / generating） |
| `sources` | 引用列表 + 回答模式 |
| `token` | 逐 token 答案文本 |
| `done` | 完整 `SearchResponse`（含 answer、citations、mode） |

---

## 2. 入库

把 Hexo **源码仓**（Markdown + PDF）灌进 Qdrant：

```bash
# 全量重建
python -m api.ingest --repo D:/Blog --recreate

# 增量：只 embed 变更的 slug
python -m api.ingest --repo D:/Blog --incremental

# 增量 + LLM 自动生成文章摘要（缺失才调用）
python -m api.ingest --repo D:/Blog --incremental --summarize
```

- Markdown 解析 frontmatter 构造 URL；PDF 抽取文本（fitz）
- 幂等 upsert：point id = `uuid5(slug:chunk_index)`，可重复运行
- 增量模式按 slug 内容 hash 比对，无变更跳过重嵌

### 线上热重载

无需重启服务即可同步最新文章：

```bash
curl -X POST http://localhost:8000/api/admin/reload \
  -H "x-admin-token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"repo":"/opt/blog","incremental":true,"summarize":true}'
```

---

## 3. 部署到服务器

```bash
# 服务器端
cd /opt/blog-rag-backend
git pull
.venv/bin/pip install -r requirements.txt
systemctl restart blog-rag
```

systemd 配置实现常驻运行 + 开机自启 + 崩溃自动拉起，Nginx 反向代理转发 `/api/*` 到 `127.0.0.1:8000`。

常用运维命令：

```bash
journalctl -u blog-rag -f     # 实时日志
systemctl status blog-rag     # 运行状态
systemctl restart blog-rag    # 重启
```

---

## 4. 评估

### 检索评估

50 条人工标注 eval 集（concept / term / howto / personal 四类），支持三模式控制变量：

```bash
python -m eval.eval_recall --mode raw     # 纯向量 baseline
python -m eval.eval_recall --mode hybrid  # 向量 + BM25 (RRF)
python -m eval.eval_recall --mode rerank  # hybrid + cross-encoder 重排
```

指标：recall@k / MRR / 分离度 Δ / 四类分诊，自动与上次结果 diff。

**实测数据**（1717 chunk / 87 slug，rerank 模式下取候选池 50）：

| 模式 | R@3 | R@5 | R@10 | MRR |
|------|-----|-----|------|-----|
| raw | 0.600 | 0.640 | 0.720 | 0.561 |
| hybrid | 0.878 | ~0.90 | 1.000 | 0.803 |
| rerank | 0.980 | ~0.98 | 1.000 | 0.871 |

### 端到端 RAGAS 评估

用生产链路跑完整生成，RAGAS 打分 Faithfulness + AnswerRelevancy：

```bash
python -m eval.eval_ragas --limit 3 --tag smoke    # 冒烟
python -m eval.eval_ragas --tag baseline           # 完整 50 条
```

结果写入 `eval/results/ragas_results_*.json`。

---

## 5. 环境变量速查

| 变量 | 说明 | 默认 |
|------|------|------|
| `QDRANT_URL` / `QDRANT_API_KEY` | 向量库连接 | — |
| `ZHIPU_API_KEY` | 智谱 API（embedding + 意图分类） | — |
| `DEEPSEEK_API_KEY` | DeepSeek API（生成） | — |
| `ADMIN_TOKEN` | `/admin/reload` 鉴权令牌 | — |
| `EMBED_MODEL` / `EMBED_BASE_URL` / `EMBED_DIM` | Embedding 配置 | embedding-3 / 智谱 / 2048 |
| `GEN_MODEL` / `GEN_BASE_URL` | 生成模型 | deepseek-chat / DeepSeek |
| `RETRIEVAL_CANDIDATE_K` | rerank 候选池大小 | 8 |
| `GENERATION_CONTEXT_K` | 最终送入上下文块数 | 5 |
| `RERANK_RELEVANCE_THRESHOLD` | 站内资料相关性门槛 | 0.30 |
| `API_KEY` / `ALLOWED_ORIGINS` / `RATE_LIMIT_PER_MIN` | API 加固 | 关 / * / 10 |
| `RERANK_BACKEND` | 重排后端（local 为 ONNX） | local |
| `WARMUP_ON_START` | 启动时预加载 BM25 + reranker | True |

---

## 6. 已知注意点

- **Embedding 维度**：换模型必须同步改 `EMBED_DIM`，否则检索失败。
- **PDF 无文本层**：需 OCR，当前仅标记占位。
- **缓存**：进程内 lru_cache（query embedding 256 / blog overview 1），多实例不共享。
- **后端不可用时**：前端自动回退到博客本地字符搜索。
