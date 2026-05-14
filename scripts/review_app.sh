#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${UV_CACHE_DIR:=/private/tmp/uv-cache}"
: "${BACKEND_URL:=http://127.0.0.1:8000}"
: "${FRONTEND_URL:=http://127.0.0.1:3000}"

export UV_CACHE_DIR

INDEX_DIR="${QMS_INDEX_DIR:-.data/qms-index}"
OPENAI_STATE="${OPENAI_VECTOR_STORE_STATE:-.data/openai/vector_store_state.json}"

info() {
  printf '[review] %s\n' "$*"
}

warn() {
  printf '[review] warning: %s\n' "$*" >&2
}

failures=0

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[review] error: required command not found: %s\n' "$1" >&2
    exit 127
  fi
}

check_url() {
  local label="$1"
  local url="$2"
  if curl --fail --silent --show-error --max-time 5 "$url" >/tmp/medai-qms-review-response.txt; then
    info "$label is reachable: $url"
  else
    warn "$label is not reachable or returned an unhealthy status: $url"
    failures=$((failures + 1))
  fi
}

require_command uv
require_command curl

info "checking repository review prerequisites"

if [ ! -f "$INDEX_DIR/manifest.json" ] || [ ! -f "$INDEX_DIR/qms.sqlite" ]; then
  warn "local index is missing or incomplete at $INDEX_DIR"
  warn "build a portable review index with: uv run ingest-qms && uv run build-qms-index --hash-embeddings"
  failures=$((failures + 1))
else
  info "local index files are present at $INDEX_DIR"
fi

if [ ! -f "$OPENAI_STATE" ]; then
  warn "OpenAI hosted File Search state is missing at $OPENAI_STATE; local fallback should be expected in review"
else
  info "OpenAI hosted File Search state is present at $OPENAI_STATE"
fi

if [ ! -f .env ] || ! grep -q '^OPENAI_API_KEY=.\+' .env; then
  warn "OPENAI_API_KEY is not set in .env; OpenAI calls may be unavailable"
  warn "for reviewer smoke tests, run backend with QMS_USE_HASH_EMBEDDINGS=true"
else
  info "OPENAI_API_KEY is present in .env"
fi

info "collecting CLI status"
if uv run search-status \
  --tasks TASKS.md \
  --index-dir "$INDEX_DIR" \
  --openai-state "$OPENAI_STATE" >/tmp/medai-qms-review-status.json; then
  info "wrote /tmp/medai-qms-review-status.json"
else
  warn "search-status failed; inspect the message above and index/env prerequisites"
  failures=$((failures + 1))
fi

info "checking running backend and frontend"
check_url "backend health" "$BACKEND_URL/health"
check_url "backend index status" "$BACKEND_URL/index/status"
check_url "frontend" "$FRONTEND_URL"

if [ "$failures" -gt 0 ]; then
  warn "$failures review check(s) need attention"
  warn "start local review servers with: scripts/dev.sh"
  exit 1
fi

info "review checks passed"
