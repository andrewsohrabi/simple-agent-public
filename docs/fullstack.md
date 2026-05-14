# Fullstack Guide

Run the MedAI QMS search API as a FastAPI server with the React search
workbench frontend. See the [core README](../README.md) for initial setup and
`docs/production-readiness-plan.md` for remaining production gaps.

## Prerequisites

- Everything in the core README
- Node.js 18+

## Start

Run both processes in separate terminals for manual local review.

**Terminal 1 — backend:**

```bash
uv run serve
```

Server starts at `http://localhost:8000`.

**Terminal 2 — frontend:**

```bash
cd frontend
npm install   # first time only
npm run dev
```

Vite prints the selected URL. The default UI URL is usually
`http://localhost:3000`.

For isolated Playwright/dev verification without colliding with another local
service, use:

```bash
HOST=127.0.0.1 PORT=8017 UVICORN_RELOAD=false QMS_USE_HASH_EMBEDDINGS=true uv run serve
```

```bash
cd frontend
VITE_API_URL=http://127.0.0.1:8017 npm run dev -- --host 127.0.0.1 --port 3017
```

That UI runs at `http://127.0.0.1:3017`.

## API

```
POST /search
Content-Type: application/json

{
  "query": "Find BOM-055 Rev G",
  "mode": "local",
  "limit": 8
}
```

```json
{
  "answer": "I found source-backed MedAI QMS evidence...",
  "citations": [],
  "query_plan": {},
  "retrieved_documents": [],
  "warnings": []
}
```

Useful status endpoints:

- `GET /health`
- `GET /stats`

`POST /chat` routes the latest user message through the same source-backed QMS
search pipeline and returns both `reply` and the structured search payload.

## How it works

`src/agent/server.py` exposes QMS search through `QmsSearchService`. The
workbench (`frontend/src/App.jsx`) calls `/stats` for model/index status and
`/search` for source-backed QMS retrieval.

## Relevant files

```
src/agent/
├── core.py          # generic agent factory
├── server.py        # FastAPI app
└── search/          # QMS ingestion, indexing, retrieval, answers, stats

frontend/
├── src/
│   ├── App.jsx      # search workbench
│   └── main.jsx     # React entry point
├── e2e/             # Playwright end-to-end tests
├── playwright.config.js
├── index.html
├── vite.config.js
└── package.json
```

## Verification

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
cd frontend
npm run build
npm run test:e2e
```

In this Codex macOS sandbox, Chromium may fail before page assertions with
`MachPortRendezvousServer ... Permission denied`. `scripts/check.sh` treats only
that exact browser-launch signature as a sandbox skip after backend startup and
frontend build pass.
