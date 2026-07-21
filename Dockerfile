# 阿里云轻量应用服务器部署用镜像
# 基础镜像：python:3.11-slim（体积小、够用）
FROM python:3.11-slim

# 环境变量（构建/运行通用）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # 国内拉取 HuggingFace 模型走镜像站（local reranker 用）
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

# 1) 先装依赖，利用 Docker 层缓存：只有 requirements.txt 变了才重装
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
 && pip install gunicorn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 再拷源码（运行时只需要 api/ core/ data/ 三个包）
COPY api/ ./api/
COPY core/ ./core/
COPY data/ ./data/

# 3) 用 gunicorn + uvicorn worker 常驻；VM 上无 250MB 限制，可直接用本地 reranker
# -w 1（单 worker）：auth._rate 限流字典与 cache._store 缓存都是进程内状态，
#   多 worker 会各持一份 → 限流阈值实际翻倍、缓存命中率减半，BM25/reranker
#   索引也会各占一份内存。个人博客流量单 worker 足够；要高并发再上 Redis 做共享。
EXPOSE 8000
CMD ["gunicorn", "api.serve:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "1", "-b", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", "--error-logfile", "-"]
