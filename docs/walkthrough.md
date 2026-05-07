# MedAI QMS Demo Walkthrough

This walkthrough reflects the current OpenAI-backed indexed baseline and calls
out the remaining production gaps separately.

## Prerequisites

- Branch: `codex/mvp`
- Corpus artifact at repo root: `Example_QMS_-_MedAI.zip`
- Python environment synced with `uv sync`
- OpenAI key available for `text-embedding-3-large` and `gpt-5.5`
- Optional native search dependencies installed when you want to exercise local
  `faiss.IndexFlatIP` and the Qwen CrossEncoder reranker instead of the
  documented fallbacks:
  `uv sync --group native-search`
- Node.js 18+ for the React frontend

Current caveat:

- The local index is now built with OpenAI `text-embedding-3-large` at 3072
  dimensions and hosted OpenAI File Search is synced.
- The optional Qwen CrossEncoder reranker path is implemented. This sandbox
  currently reports `deterministic_fallback` because `sentence_transformers` and
  the local reranker model cache are not installed here.
- The optional native FAISS path is implemented. If `faiss` is unavailable,
  local vector search uses a NumPy `IndexFlatIP`-compatible fallback and reports
  that backend in status/debug output.

Check status:

```bash
git branch --show-current
git status --short
uv run pytest evals/ -v
```

## 1. Build Or Verify The Current OpenAI Index

```bash
uv run ingest-qms
uv run build-qms-index
```

Expected output should include:

- Corpus hash.
- Document count `189`, skipped count `0`, metadata-only count `24`, and chunk
  count `7,778` for the current local index.
- Embedding model: `text-embedding-3-large`.
- Embedding dimensions: `3072`.
- Embedding provider: `openai`.
- FAISS type: `IndexFlatIP`.
- SQLite `revisions` and `doc_references`; current status reports `2,355`
  references.
- Hosted File Search status `synced` with `189` files for the current corpus
  hash.
- Vector backend: `faiss` when the optional native package is importable,
  otherwise `numpy_fallback`.
- Reranker backend: `sentence_transformers_cross_encoder` when the configured
  CrossEncoder model is available, otherwise `deterministic_fallback` with a
  warning.
- Chunking: 600-token child chunks, 100-token overlap, 700-token table target,
  metadata chunks enabled, and one answer-time neighbor chunk.
- Extraction warnings.

Inspect status:

```bash
uv run search-status --tasks TASKS.md --index-dir .data/qms-index --openai-state .data/openai/vector_store_state.json
```

Deterministic smoke build, only when live embeddings are intentionally avoided:

```bash
uv run ingest-qms
uv run build-qms-index --hash-embeddings
```

The OpenAI build must remain the indexed baseline for production-style runs.

## 2. Smoke-Test Retrieval

```bash
uv run search-evals --dataset evals/datasets/qms_smoke.jsonl --report docs/eval-runs --mode local
```

Expected behavior:

- Returns `RSK-P01-016 - MX1 MedAI PFMEA_A-Signed.docx` near the top.
- Shows evidence scores and source metadata.
- Marks signed/active status when parsed.

Additional smoke queries:

- `What is the latest active BOM-055 revision?`
- `Find verification reports for MX1 software system v3.3.0.`
- `Which documents mention the MedAI Rest Server?`
- `List obsolete MX1 software planning or anomaly documents.`
- `Which records support workstation installation qualification?`

## 3. Run The API

Target command:

```bash
uv run serve
```

Expected backend URL: `http://localhost:8000`

Useful checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

Target search request:

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which verification reports cover MX1 software system v3.0.0?","top_k":8}'
```

## 4. Run The Frontend

```bash
cd frontend
npm install
npm run dev
```

Expected frontend URL: `http://localhost:3000`

The first screen should be the search workbench. It should show index readiness
and provide a desktop-first chat/search entry point without requiring a separate
landing page.

## 5. Demo Script

Use this sequence for a live walkthrough:

1. Show `GET /index/status` or the UI status indicator.
2. Ask a known-item question: `Find BOM-055 Rev G.`
3. Ask a synthesis question: `Summarize the evidence for MX1 software system
   verification around v3.3.0.`
4. Ask a revision question: `Which MX1 software planning documents are obsolete,
   and what active records appear related?`
5. Ask a cross-reference question: `Connect the PFMEA to verification or
   validation evidence.`
6. Ask a counting question: `How many VVPR documents are in the corpus?`
7. Open citations and verify each cited filename/chunk maps to the answer.

## 6. Expected Answer Behavior

Good answers:

- Cite every sourced QMS fact.
- Prefer latest active revisions unless the question asks for obsolete history.
- Say when evidence is metadata-only or extraction is limited.
- Avoid claims that are not present in retrieved evidence.
- Keep source lists compact and inspectable.

Bad answers:

- Cite filenames not returned by retrieval.
- Treat obsolete records as active without warning.
- Use model knowledge about medical devices instead of corpus evidence.
- Give exact counts without explaining whether they come from filenames,
  document records, chunks, or extracted text.

## 7. Eval Handoff

After the walkthrough, record an eval run:

```bash
mkdir -p docs/eval-runs
```

Create `docs/eval-runs/YYYY-MM-DD-medai-qms-search.md` with:

- Git SHA and branch.
- Corpus SHA-256.
- Index manifest summary.
- Model defaults and any deviations.
- Retrieval metrics.
- Answer/citation metrics.
- Known failures or degraded modes.

## 8. Final Browser Verification

Playwright MCP/browser verification was run on the desktop workbench after the
OpenAI index, hosted/local retrieval modes, citation flow, and frontend source
inspection workflow were in place. The MCP check loaded
`http://127.0.0.1:3060`, ran the engineering-change-request enumeration example,
opened Sources and Debug, and reported zero browser console warnings/errors.

Command-line Playwright remains in the repo for normal environments. In this
Codex macOS sandbox, `scripts/check.sh` skips only the documented Chromium
MachPort permission failure after backend startup and frontend build pass.
