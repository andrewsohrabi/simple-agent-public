# MedAI QMS Demo Walkthrough

This is the target walkthrough for the completed search MVP. Commands marked as
target commands may need to be adjusted by the implementation worker if final
script names differ.

## Prerequisites

- Branch: `codex/mvp`
- Corpus artifact at repo root: `Example_QMS_-_MedAI.zip`
- Python environment synced with `uv sync`
- OpenAI key available for `text-embedding-3-large` and `gpt-5.5`
- Qwen reranker available locally or through the selected hosted endpoint
- Node.js 18+ for the React frontend

Check status:

```bash
git branch --show-current
git status --short
uv run pytest evals/ -v
```

## 1. Build The Index

```bash
uv run ingest-qms
uv run build-qms-index --hash-embeddings
```

Expected output should include:

- Corpus hash.
- Document count `189`, skipped count `0`, metadata-only count `24`, and chunk
  count `7,778` for the current deterministic local review artifact.
- Embedding model: `text-embedding-3-large`.
- Embedding dimensions: `3072`.
- FAISS type: `IndexFlatIP`.
- Chunking: 600-token child chunks, 100-token overlap, 700-token table target,
  metadata chunks enabled, and one answer-time neighbor chunk.
- Extraction warnings.

Inspect status:

```bash
uv run search-status --tasks TASKS.md --index-dir .data/qms-index --openai-state .data/openai/vector_store_state.json
```

## 2. Smoke-Test Retrieval

```bash
uv run search-evals --dataset evals/datasets/qms_smoke.jsonl --report docs/eval-runs --hash-embeddings --mode local
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
and provide a chat/search entry point without requiring a separate landing page.

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
