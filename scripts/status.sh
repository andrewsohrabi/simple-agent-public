#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${UV_CACHE_DIR:=/private/tmp/uv-cache}"
export UV_CACHE_DIR

INDEX_DIR="${QMS_INDEX_DIR:-.data/qms-index}"
OPENAI_STATE="${OPENAI_VECTOR_STORE_STATE:-.data/openai/vector_store_state.json}"

info() {
  printf '[status] %s\n' "$*"
}

warn() {
  printf '[status] warning: %s\n' "$*" >&2
}

if ! command -v uv >/dev/null 2>&1; then
  printf '[status] error: uv is required but was not found on PATH.\n' >&2
  exit 127
fi

if [ ! -f "$INDEX_DIR/manifest.json" ] || [ ! -f "$INDEX_DIR/qms.sqlite" ]; then
  warn "local index is missing or incomplete at $INDEX_DIR"
  warn "build a local review index with: uv run ingest-qms && uv run build-qms-index --hash-embeddings"
fi

if [ ! -f "$OPENAI_STATE" ]; then
  warn "OpenAI hosted File Search state is missing at $OPENAI_STATE; review will use local index fallback when available"
fi

if [ ! -f .env ] || ! grep -q '^OPENAI_API_KEY=.\+' .env; then
  warn "OPENAI_API_KEY is not set in .env; OpenAI calls may be unavailable and local hash fallback is expected"
fi

info "collecting QMS search status"
uv run search-status \
  --tasks TASKS.md \
  --index-dir "$INDEX_DIR" \
  --openai-state "$OPENAI_STATE"
