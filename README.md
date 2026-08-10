# Blog RAG Search Backend

为个人 Hexo 博客提供语义检索与 AI 问答能力的后端服务。前端为右下角悬浮 AI 聊天窗。

**架构链路**：`query → 规则快路(问候/实时零成本短路) → [LLM分类 ∥ 检索并行] → 向量+BM25(RRF融合) → 本地ONNX重排 → DeepSeek生成 → SSE流式输出`

- 后端：FastAPI（systemd 托管，Nginx 反代）
- 向量库：本地 Faiss（自托管，零网络，2048 维，内积=余弦）
- Embedding：智谱 embedding-3 / 生成：DeepSeek-chat / 意图分类：glm-4-flash
- 重排：bge-reranker-base ONNX int8 量化，CPU 本地推理，无网络依赖
- 检索：Dense + BM25 (jieba) → RRF(k=60) 融合 → cross-encoder 重排
- 前端：原生 JS，按 stage/sources/token/done 四类 SSE 事件增量渲染

---

## 1. 本地开发

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  / macOS/Linux: source .venv/bin/activate
# 自托管 / 本地开发需安装 dev 依赖集：在 requirements.txt（API 运行时）基础上，
# 额外包含入库所需依赖（python-frontmatter / pymupdf / pdfplumber / pypinyin / requests）。
# 仅纯 Vercel 部署才只用 requirements.txt（它不运行入库）。
pip install -r requirements-dev.txt
cp .env.example .env   # 填入 智谱 / DeepSeek 密钥（Faiss 本地存储，无需向量库账号）
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

把 Hexo **源码仓**（Markdown + PDF）灌进本地 Faiss：
Faiss是用于高效相似性搜索和稠密向量聚类的算法库，不是数据库，无法提供数据持久化，CRUD操作，元数据过滤和权限管理等，通过索引和压缩解决高维空间下暴力搜索速度过慢的问题。
持久化：faiss.write_index() 生成的 .bin 文件
数据加载：faiss.read_index() 读到内存
增删改查：修改元数据表，向量库需重建

```bash
# 全量重建
python -m api.ingest --repo D:/Blog --recreate

# 增量：只 embed 变更的 slug
python -m api.ingest --repo D:/Blog --incremental

# 增量 + LLM 自动生成文章摘要（缺失才调用）
python -m api.ingest --repo D:/Blog --incremental --summarize
```

- Markdown 解析 frontmatter 构造 URL；PDF 抽取文本（fitz）
- 幂等 upsert：以 `(slug, chunk_index)` 为键合并，已存在则覆盖，可重复运行
- 增量模式按 slug 内容 hash 比对，无变更跳过重嵌

### 线上热重载

无需重启服务即可同步最新文章。`/admin/reload` 会先 `git pull` 博客仓库，再增量入库并刷新 BM25：

```bash
# 直连 uvicorn 无 /api 前缀（root_path 只影响文档 URL）；经 Nginx 则是 /api/admin/reload
# repo 可省略，缺省读配置里的 BLOG_REPO_PATH
curl -X POST http://localhost:8000/admin/reload \
  -H "x-admin-token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"incremental":true,"summarize":true}'
```

### 文章发布 → 自动入库链路

内容真值在 GitHub 博客仓库，向量库真值在服务器。本地只负责写文章并 push，服务器持有一份克隆副本自行入库，**不传二进制索引**。

```
本地写 md ──push──> GitHub 博客仓库 ──Actions──> hexo generate + 部署站点
                                                        │
                                          部署完成后 curl 通知
                                                        v
                          服务器 /admin/reload：git pull → 增量入库 → 刷新 BM25
                                                        ^
                                   cron 每 10 分钟兜底（scripts/sync_blog.sh）
```

**为什么通知放在部署之后**：文章 URL 取自线上 `search.xml`，由 Actions 构建生成。若在部署前入库，新文章会「能检索但没有阅读链接」。等部署完再通知，URL 天然齐全；即使某次早了，URL 变化会让内容 hash 变，下一轮增量自动重嵌补全。

一次性配置：

| 位置 | 配置 |
|------|------|
| 服务器 | `git clone git@github.com:<user>/<blog>.git <博客目录>`（私有仓库用 SSH key；目录任选，与下面 `BLOG_REPO_PATH` 一致即可） |
| 服务器 `.env` | `BLOG_REPO_PATH=<博客目录>` 与 `ADMIN_TOKEN=<令牌>` |
| 服务器 cron | `*/10 * * * * <后端目录>/scripts/sync_blog.sh >> /var/log/rag-sync.log 2>&1` |
| 博客仓库 Secret | `RAG_ADMIN_TOKEN` = 服务器 admin 令牌（Settings → Secrets and variables → Actions） |
| 博客仓库 Variable（可选） | `RAG_API_BASE`，默认 `https://rag.zyydgrbk.top/api` |

`sync_blog.sh` 会自定位后端仓库根目录并读取同级 `.env` 里的 `BLOG_REPO_PATH` / `ADMIN_TOKEN`，**因此 cron 行里不需要写令牌**（写进 crontab 会被 `crontab -l`、`ps` 看到）。需要临时覆盖时，仍可用环境变量：`BLOG_REPO=... BACKEND_URL=... ADMIN_TOKEN=... ./scripts/sync_blog.sh`（优先级：环境变量 > `.env` > 内置默认）。

博客仓库 `.github/workflows/autodeploy.yml` 末尾已加「通知 RAG 增量入库」步骤，标了 `continue-on-error`，通知失败不影响站点部署。

---

## 3. 部署到服务器

### 3.1 基本发布流程（代码更新后）

```bash
# 服务器端
cd /opt/blog-rag-backend
git pull
# 自托管服务器既 serve 又运行 api.ingest 入库，必须安装 dev 依赖集：
# python-frontmatter / pymupdf / pdfplumber 等入库依赖均在 requirements-dev.txt，
# requirements.txt 是 Vercel 最小集，不含这些。漏装会导致入库静默 0 chunk。
.venv/bin/pip install -r requirements-dev.txt   # 依赖无变更时秒过，有变更才真正安装
systemctl restart blog-rag                  # 必须重启：systemd 常驻进程启动即把代码载入内存，git pull 只改磁盘文件
```
systemd 配置实现常驻运行 + 开机自启 + 崩溃自动拉起，Nginx 反向代理转发 `/api/*` 到 `127.0.0.1:8000`。

常用运维命令：

```bash
journalctl -u blog-rag -f     # 实时日志
systemctl status blog-rag     # 运行状态
systemctl restart blog-rag    # 重启
```

### 3.2 备案后上线

国内服务器对外提供 Web 服务前必须完成 **ICP 备案**；备案通过后，才能把域名指向本服务器并启用 HTTPS。本项目的切流链路：

```
用户浏览器 → https://rag.zyydgrbk.top/api/search
                ↓（Cloudflare DNS 仅解析 / 灰云）
            Nginx（监听 443，终止 TLS + 反代）
                ↓
            本机 uvicorn（127.0.0.1:8000，FastAPI，root_path=/api）
```

### 3.3 DNS 托管（Cloudflare）

- 域名 NS 已委派给 Cloudflare（`*.ns.cloudflare.com`），因此**阿里云云解析 DNS 的记录不生效**，所有子域记录都在 Cloudflare 后台添加。
- 添加记录：类型 `A`、名称 `rag`、IPv4 `47.116.188.225`、代理状态 **灰色云朵（仅 DNS）**。
  - 灰色云朵：用户直连阿里云服务器，路径最短、最稳，且符合备案要求。
  - 橙色云朵（代理）：流量先经 Cloudflare 再回源，存在备案合规风险，且可能影响 SSE 流式响应；如坚持使用见 3.5 注。
- 生效验证：`nslookup rag.zyydgrbk.top` 应返回 `47.116.188.225`。
- 阿里云安全组（防火墙）放行 **80 / 443** 入站，否则证书申请与访问都会失败。

### 3.4 Nginx 反代 + HTTPS 证书（certbot）

Nginx 是服务器上独立运行的程序（非 Python 库），角色是 HTTPS 终止 + 反向代理。系统：Ubuntu 24.04。

```bash
# 1. 安装 Nginx + certbot
sudo apt update && sudo apt install -y nginx certbot python3-certbot-nginx

# 2. 删默认站点，写入「引导配置」（仅 80 端口，供 certbot 做 HTTP 验证）
sudo rm -f /etc/nginx/sites-enabled/default
sudo tee /etc/nginx/sites-available/rag.zyydgrbk.top > /dev/null <<'EOF'
server {
    listen 80;
    listen [::]:80;
    server_name rag.zyydgrbk.top;
    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 120s;
    }
    location / { return 301 https://$host$request_uri; }
}
EOF
sudo ln -s /etc/nginx/sites-available/rag.zyydgrbk.top /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 3. 申请 Let's Encrypt 证书（把邮箱换成你自己的）
sudo certbot certonly --nginx -d rag.zyydgrbk.top \
  --non-interactive --agree-tos --email you@example.com --no-eff-email

# 4. 覆盖为「最终配置」（443 + SSL + SSE 关缓冲）
sudo tee /etc/nginx/sites-available/rag.zyydgrbk.top > /dev/null <<'EOF'
server {
    listen 80;
    listen [::]:80;
    server_name rag.zyydgrbk.top;
    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 301 https://$host$request_uri; }
}
server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name rag.zyydgrbk.top;
    ssl_certificate     /etc/letsencrypt/live/rag.zyydgrbk.top/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/rag.zyydgrbk.top/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
    add_header Strict-Transport-Security "max-age=31536000" always;
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection "";
        chunked_transfer_encoding on;
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
    }
}
EOF
sudo nginx -t && sudo systemctl reload nginx
```

### 3.5 验证与收尾

```bash
curl https://rag.zyydgrbk.top/api/health            # 应返回 {"status":"ok"}（证书受信任，无需 -k）
curl -k -X POST https://rag.zyydgrbk.top/api/search \
  -H "Content-Type: application/json" -d '{"query":"你的博客是怎么搭建的","top_k":3}'
```

- 前端 `rag-client.js` 生产端点指向 `https://rag.zyydgrbk.top/api/search`；若博客 `_config.butterfly.yml` 用 `window.RAG_SEARCH_ENDPOINT` 覆盖，**务必带 `/api`**，否则请求落不到 Nginx `/api/` 反代、CORS 预检失败。
- 后端 `API_KEY` 留空即放行（前端不放密钥，靠 CORS + 限流防滥用）；`ALLOWED_ORIGINS` 建议收紧为 `https://zyydgrbk.top,http://localhost:4000`。

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
| `FAISS_INDEX_PATH` / `FAISS_META_PATH` | 本地 Faiss 索引与元数据路径 | data/vector_store.* |
| `BLOG_REPO_PATH` | 服务器侧博客仓库克隆路径，`/admin/reload` 缺省从这里 pull 并入库 | — |
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
