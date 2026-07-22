# Blog RAG Search Backend

为个人 Hexo + Butterfly 博客增加**语义检索 / AI 问答**能力的后端服务。

- 后端：FastAPI，可部署为 Vercel serverless 函数
- 向量库：Qdrant Cloud 免费层
- 生成：DeepSeek-chat ｜ Embedding：智谱 embedding-3（2048 维）
- 重排：Jina Reranker v2 multilingual API
- 前端：右下角悬浮 AI 聊天窗 + 旧字符搜索兜底

> 架构设计见 `D:\wiki\博客RAG检索功能-架构分析v2-20260711.md`

---

## 1. 本地开发

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
copy .env.example .env   # Windows；填入你的 key / Qdrant 地址
python -m uvicorn api.serve:app --reload --port 8000
```

健康检查与搜索：

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"如何部署博客","top_k":5}'
```

### 流式搜索

`POST /search/stream` 使用 `text/event-stream` 返回事件。事件顺序为：

- `sources`：引用列表和回答模式；
- 一个或多个 `token`：`{"text": "..."}`，可直接增量显示；
- 可选 `error`：生成失败的说明；
- `done`：完整 `SearchResponse`，其中包含 `answer`、`citations`、`fallback` 和 `mode`。

`/search` 与 `/search/stream` 共享基于 `query + top_k` 的进程内缓存。前端默认优先消费流式接口；若流式请求不可用则回退至 JSON 接口和博客本地字符搜索。

## 2. 入库（Ingestion）

把 Hexo **源码仓**（Markdown + 内嵌 PDF 本地仓库）灌进 Qdrant：

```bash
python -m api.ingest --repo D:/Blog
```

- Markdown 用 frontmatter 构造文章 URL；PDF 抽取文本（无文本层标记 `ocr`）。
- 幂等：point id = `slug:chunk_index`，可重复运行。
- 当前仅在源码仓有内容时入库；CI 触发见 `scripts/ingest_ci.sh`。

## 3. 部署到 Vercel
vercel站点托管平台：支持部署serverless接口，不仅可以部署静态网站，还可以部署动态网站，只需要自己写函数/接口，Vercel在请求来时临时拉起运行环境执行
在vercel.json中进行vercel部署配置

安装：npm i -g vercel
登录：vercel login
链接：vercel link

```bash
# 把vercel后台配置的环境变量拉到本地文件
vercel env pull .env.production.local
# 发版部署 --prod部署到生产地址
vercel deploy
```

`vercel.json` 已把 `/search`、`/search/stream` 与 `/health` 指向 `api/index.py`。
部署后把前端 `RAG_SEARCH_ENDPOINT` 设为你的 Vercel 函数地址。

vercel地址：https://blog-rag-backend.vercel.app

## 4. 前端注入（Hexo + Butterfly）

见 `frontend/butterfly-inject.md`：把 `rag-client.js` / `rag-search.css` 放进
博客源码仓，并在 `layout.ejs`（或 injector 的 `bottom`）引入即可出现悬浮窗。
旧 `search.xml` 字符搜索保留，AI 失败时自动回退。

## 5. 环境变量速查

| 变量 | 说明 | 默认 |
|---|---|---|
| `QDRANT_URL` / `QDRANT_API_KEY` | 向量库 | — |
| `EMBED_MODEL` / `EMBED_BASE_URL` / `EMBED_DIM` | Embedding | embedding-3 / 智谱 / 2048 |
| `EMBED_BATCH_SIZE` / `QDRANT_UPSERT_BATCH_SIZE` / `QDRANT_WRITE_TIMEOUT` | 入库批处理与写超时 | 64 / 32 / 120 |
| `QDRANT_READ_TIMEOUT` / `QDRANT_WARMUP_ENABLED` | 查询超时 / 启动时预热查询连接 | 3 / 开 |
| `GEN_MODEL` / `GEN_BASE_URL` / `DEEPSEEK_API_KEY` | 生成 | deepseek-chat |
| `JINA_API_KEY` | Jina Reranker v2 重排 | — |
| `RETRIEVAL_CANDIDATE_K` / `GENERATION_CONTEXT_K` / `RERANK_RELEVANCE_THRESHOLD` | rerank 候选池 / 最终上下文块数 / 站内资料相关性门槛 | 8 / 3 / 0.30 |
| `API_KEY` / `ALLOWED_ORIGINS` / `RATE_LIMIT_PER_MIN` / `CACHE_TTL` | 加固 | 关 / * / 10 / 600 |

## 6. 测试

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

默认测试用 mock 替代 LLM / Embedding / Qdrant / 联网，无需真实 key；带 `integration` 标记的真实链路测试会跳过。需要验证真实服务时显式开启：

```powershell
$env:RUN_INTEGRATION = "1"
.venv\Scripts\python.exe -m pytest tests -q -m integration
```

重复端到端性能测试会写入 `.codex/PERF_REPORT.md`：

```powershell
$env:RUN_INTEGRATION = "1"
.venv\Scripts\python.exe scripts/perf_bench.py --runs 5
```

## 7. 已知注意点

- **国内访问**：Vercel + Qdrant(EU) 对国内用户可能偏慢，必要时改国内部署
  （阿里云 FC / 腾讯云 SCF）。
- **Embedding 维度**：换 embedding 模型必须同步改 `EMBED_DIM`，否则建库/检索失败。
- **PDF 无文本层**：需 OCR 才能检索，当前仅标记 `ocr` 占位。
- **缓存**：serverless 下为进程内缓存，多实例不共享；生产可换 Redis/KV。
- **外部依赖降级**：Qdrant 或模型服务不可用时，接口返回 `fallback: true`；前端应保留本地搜索兜底。
- **性能数据**：性能脚本不走 HTTP 缓存且会调用真实服务；比较前后结果时请保持模型、数据集、区域和查询不变。

查询主链路深拆
1、前端 → FastAPI（api/serve.py）






## pdf优化

决策选型:
| 特性 | fitz (PyMuPDF) | pdfplumber |
| :--- | :--- | :--- |
| **速度** | 极快（C 底层） | 慢 3-5x（纯 Python 解析） |
| **文本质量** | 直接拿 PDF 文字流，保留换行结构 | 额外做字符位置分析，表格/列提取更准 |
| **适合场景** | 普通文章/技术文档 PDF | 含表格、多栏布局的 PDF |
| **问题** | 多栏 PDF 可能混行 | 偶尔空格/换行过多 |

split_text（保持） vs split_documents
split_text(text: str) — 输入纯字符串，返回 List[str]
split_documents(docs: List[Document]) — 输入 LangChain Document 对象列表（带 metadata），返回 List[Document]

添加中文优先的分隔符顺序：段落→句子→子句→字符
_ZH_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", ""]
在 frontmatter 加 permalink: /your-slug/，一旦设了就永远不变，代码里 _build_url 会优先用它


嵌入模型：
智谱embedding-3：内容80%中文+代码+短英文术语，维度2048维度已经足够
bce-embedding-base_v1（百川，中英双语很强，后续再说）

将pdf按照标题进行切分：
匹配 Markdown 风格标题（# / ## / ###）或全大写短行（≤60字符，视为章节标题）

清理pdf提取噪音：_clean_text（页眉页脚、连续空行、控制字符）
目前pdf的slug使用的是标题名：
slug = os.path.splitext(fname)[0]

DocumentChunk统一数据结构：下游完全不需要知道内容来自哪里
slug时Qdrant里每个点的逻辑主键，用于生成确定性UUID：
id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{c.slug}:{c.chunk_index}")
表示：
- 同一篇文章重新 ingest，生成的 UUID 不变 → upsert 幂等，不会重复写入
- title 可以改，但 slug 不变的话，同一篇文章的点会被原地更新而不是新增

title不够，可以重复，但是slug从文件相对路径派生的，天然唯一

url 是该 chunk 对应的博客可访问链接，用于 RAG 回答时给用户附上"来源链接"。

生成逻辑优先级（parse_hexo.py:L84-L120）：

frontmatter 里有 permalink 或 url → 直接用
有日期的 post → {SITE_URL}/YYYY/MM/DD/{pinyin_slug}/
没日期的 page → {SITE_URL}/{pinyin_slug}/
关于链接时间是否会变动：Hexo 的 permalink 默认格式是 :year/:month/:day/:title/，时间取的是 frontmatter 里的 date 字段，不是文件修改时间。所以只要你不改 frontmatter 的 date，链接就不会变。如果你改了内容但没改日期，链接稳定。

如果你担心不一致，最保险的方式是在 frontmatter 里显式写 permalink: /your-custom-path/，这样代码会优先用它，完全不受日期影响。

验收脚本用法
导入完成后运行：


# 完整验收（数量 + payload + URL样本 + 检索效果）
python -m scripts.verify_ingest --repo D:/Blog

# 只验数量和字段，不调嵌入 API（省钱）
python -m scripts.verify_ingest --repo D:/Blog --skip-retrieval
脚本输出四个部分：

数量对比 — 本地解析 N 条 vs Qdrant 存了 N 条，差值显示是否有漏写
Payload 完整性 — 按 post/page/pdf 各抽 5 条，逐字段检查
URL 样本 — 打出每类前 5 篇的 slug + url，你对照浏览器里的实际链接确认
检索效果 — 6 个预设查询，每条显示 top3 命中标题/URL/内容片段和相似度分（✓ ≥0.50 / ~ ≥0.40 / ✗ <0.40）
