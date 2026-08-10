#!/usr/bin/env bash
set -euo pipefail

# 脚本自定位：无论从哪个目录被 cron 调用，都能找到后端仓库根目录与 .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$BACKEND_DIR/.env}"

# 从 .env 读取单个键值。这样令牌只存在于 .env 一处，
# 不必写进 crontab（crontab -l / ps 都能看到，等于明文泄露）。
read_env() {
  [ -f "$ENV_FILE" ] || return 0
  local line
  line="$(grep -m1 -E "^[[:space:]]*$1[[:space:]]*=" "$ENV_FILE" || true)"
  [ -n "$line" ] || return 0
  line="${line#*=}"
  line="${line#"${line%%[![:space:]]*}"}"   # 去左空白
  line="${line%"${line##*[![:space:]]}"}"   # 去右空白
  line="${line%\"}"; line="${line#\"}"      # 去成对双引号
  line="${line%\'}"; line="${line#\'}"      # 去成对单引号
  printf '%s' "$line"
}

# ===== 三个值的优先级：显式环境变量 > .env > 内置默认 =====
BLOG_REPO="${BLOG_REPO:-$(read_env BLOG_REPO_PATH)}"   # Hexo 源仓库（含 source/）
BLOG_REPO="${BLOG_REPO:-/opt/Hexo}"
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"    # 直连 uvicorn，无 /api 前缀
ADMIN_TOKEN="${ADMIN_TOKEN:-$(read_env ADMIN_TOKEN)}"  # 默认从 .env 读，无需写进 crontab

if [ ! -d "$BLOG_REPO/.git" ]; then
  echo "ERROR: 博客仓库不存在或不是 git 仓库: $BLOG_REPO" >&2
  echo "       请先 git clone，或在 $ENV_FILE 里设置 BLOG_REPO_PATH=<实际路径>" >&2
  exit 1
fi

echo "==> 1/3 pull blog repo"
git -C "$BLOG_REPO" pull --ff-only

echo "==> 2/3 trigger incremental ingest + BM25 refresh via /admin/reload"
curl -fsS -X POST "$BACKEND_URL/admin/reload" \
  -H "Content-Type: application/json" \
  ${ADMIN_TOKEN:+-H "x-admin-token: $ADMIN_TOKEN"} \
  -d "{\"repo\": \"$BLOG_REPO\", \"incremental\": true, \"summarize\": true}"

echo "==> done"
