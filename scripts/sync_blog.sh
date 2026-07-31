#!/usr/bin/env bash
set -euo pipefail

# ===== 按你服务器实际路径改 =====
BLOG_REPO="/opt/blog"                 # Hexo 源仓库（含 source/）
BACKEND_URL="http://localhost:8000"   # 本地后端走 /admin/reload（无 /api 前缀）
ADMIN_TOKEN="${ADMIN_TOKEN:-}"        # 从环境变量读，别硬编码进脚本

echo "==> 1/3 pull blog repo"
git -C "$BLOG_REPO" pull --ff-only

echo "==> 2/3 trigger incremental ingest + BM25 refresh via /admin/reload"
curl -fsS -X POST "$BACKEND_URL/admin/reload" \
  -H "Content-Type: application/json" \
  ${ADMIN_TOKEN:+-H "x-admin-token: $ADMIN_TOKEN"} \
  -d "{\"repo\": \"$BLOG_REPO\", \"incremental\": true, \"summarize\": true}"

echo "==> done"
