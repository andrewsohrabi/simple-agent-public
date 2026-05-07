#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${UV_CACHE_DIR:=/private/tmp/uv-cache}"
export UV_CACHE_DIR

INDEX_DIR="${QMS_INDEX_DIR:-.data/qms-index}"
OPENAI_STATE="${OPENAI_VECTOR_STORE_STATE:-.data/openai/vector_store_state.json}"

info() {
  printf '[check] %s\n' "$*"
}

warn() {
  printf '[check] warning: %s\n' "$*" >&2
}

if ! command -v uv >/dev/null 2>&1; then
  printf '[check] error: uv is required but was not found on PATH.\n' >&2
  exit 127
fi

if [ ! -f "$INDEX_DIR/manifest.json" ] || [ ! -f "$INDEX_DIR/qms.sqlite" ]; then
  warn "local index is missing or incomplete at $INDEX_DIR"
  warn "status checks will report degraded/missing index; build with: uv run ingest-qms && uv run build-qms-index --hash-embeddings"
fi

if [ ! -f "$OPENAI_STATE" ]; then
  warn "OpenAI hosted File Search state is missing at $OPENAI_STATE; local fallback is expected"
fi

if [ ! -f .env ] || ! grep -q '^OPENAI_API_KEY=.\+' .env; then
  warn "OPENAI_API_KEY is not set in .env; OpenAI-backed paths may be unavailable"
fi

info "running Python tests"
uv run pytest -q

info "collecting status snapshot"
uv run search-status \
  --tasks TASKS.md \
  --index-dir "$INDEX_DIR" \
  --openai-state "$OPENAI_STATE" >/tmp/medai-qms-search-status.json
info "wrote /tmp/medai-qms-search-status.json"

if [ -d frontend ]; then
  (
    cd frontend
    if [ ! -d node_modules ]; then
      warn "frontend/node_modules is missing; running npm install before frontend checks"
      npm install
    fi
    info "building frontend"
    npm run build
    info "running frontend E2E checks"
    export PLAYWRIGHT_API_PORT="${PLAYWRIGHT_API_PORT:-$((18000 + ($$ % 1000)))}"
    export PLAYWRIGHT_FRONTEND_PORT="${PLAYWRIGHT_FRONTEND_PORT:-$((19000 + ($$ % 1000)))}"
    info "using Playwright ports api=$PLAYWRIGHT_API_PORT frontend=$PLAYWRIGHT_FRONTEND_PORT"
    e2e_log="$(mktemp -t medai-qms-playwright.XXXXXX.log)"
    set +e
    npm run test:e2e 2>&1 | tee "$e2e_log"
    e2e_status="${PIPESTATUS[0]}"
    set -e
    if [ "$e2e_status" -ne 0 ]; then
      if grep -q 'MachPortRendezvousServer' "$e2e_log" && grep -q 'Permission denied' "$e2e_log"; then
        warn "skipping frontend E2E because sandboxed macOS Chromium cannot register MachPort; build and backend startup already passed"
      else
        warn "frontend E2E failed; log retained at $e2e_log"
        exit "$e2e_status"
      fi
    fi
  )
fi
