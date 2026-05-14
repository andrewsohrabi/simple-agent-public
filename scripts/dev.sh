#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${UV_CACHE_DIR:=/private/tmp/uv-cache}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${FRONTEND_HOST:=127.0.0.1}"
: "${FRONTEND_PORT:=3000}"
: "${UVICORN_RELOAD:=true}"

export UV_CACHE_DIR HOST PORT UVICORN_RELOAD
export VITE_API_URL="${VITE_API_URL:-http://$HOST:$PORT}"

INDEX_DIR="${QMS_INDEX_DIR:-.data/qms-index}"

info() {
  printf '[dev] %s\n' "$*"
}

warn() {
  printf '[dev] warning: %s\n' "$*" >&2
}

cleanup() {
  if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! command -v uv >/dev/null 2>&1; then
  printf '[dev] error: uv is required but was not found on PATH.\n' >&2
  exit 127
fi

if ! command -v npm >/dev/null 2>&1; then
  printf '[dev] error: npm is required for the frontend but was not found on PATH.\n' >&2
  exit 127
fi

if ! command -v curl >/dev/null 2>&1; then
  printf '[dev] error: curl is required for server readiness checks but was not found on PATH.\n' >&2
  exit 127
fi

wait_for_url() {
  local label="$1"
  local url="$2"
  local pid="$3"
  local attempts=30

  for _ in $(seq 1 "$attempts"); do
    if curl --fail --silent --max-time 2 "$url" >/dev/null 2>&1; then
      info "$label is ready: $url"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      warn "$label process exited before becoming ready"
      return 1
    fi
    sleep 1
  done

  warn "$label did not become reachable at $url"
  return 1
}

if [ ! -f "$INDEX_DIR/manifest.json" ] || [ ! -f "$INDEX_DIR/qms.sqlite" ]; then
  warn "local index is missing or incomplete at $INDEX_DIR"
  warn "build it first with: uv run ingest-qms && uv run build-qms-index --hash-embeddings"
fi

if [ ! -f .env ] || ! grep -q '^OPENAI_API_KEY=.\+' .env; then
  warn "OPENAI_API_KEY is not set in .env; enabling local hash embedding fallback for dev"
  export QMS_USE_HASH_EMBEDDINGS="${QMS_USE_HASH_EMBEDDINGS:-true}"
else
  info "OPENAI_API_KEY is present in .env; backend will use configured provider unless overridden"
fi

if [ ! -f .data/openai/vector_store_state.json ]; then
  warn "OpenAI hosted File Search state is unavailable; backend should fall back to local search when possible"
fi

if [ ! -d frontend/node_modules ]; then
  info "installing frontend dependencies"
  (cd frontend && npm install)
fi

info "starting backend at http://$HOST:$PORT"
uv run serve &
BACKEND_PID=$!
if ! wait_for_url "backend health" "http://$HOST:$PORT/health" "$BACKEND_PID"; then
  warn "backend is down or a different service is bound to $HOST:$PORT"
  exit 1
fi

info "starting frontend at http://$FRONTEND_HOST:$FRONTEND_PORT"
(
  cd frontend
  npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" --strictPort
) &
FRONTEND_PID=$!
if ! wait_for_url "frontend" "http://$FRONTEND_HOST:$FRONTEND_PORT" "$FRONTEND_PID"; then
  warn "frontend is down or port $FRONTEND_PORT is unavailable"
  exit 1
fi

info "dev servers are running; press Ctrl-C to stop"
wait "$BACKEND_PID" "$FRONTEND_PID"
