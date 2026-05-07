# Production Readiness Plan

This document reconciles the current `codex/mvp` implementation against the
original MedAI QMS internal-search plan. It is intentionally direct: the current
branch has crossed the OpenAI index, hosted File Search, local/hosted retrieval,
optional real reranking, guarded `gpt-5.5` answer synthesis, API, desktop UI,
eval-reporting, and browser-verification milestones. The remaining quality gap
is retrieval/evidence precision and broader quality eval improvement, not a
missing core technical component.

## Current Baseline

- Branch: `codex/mvp`
- Current demo mode: local and hosted QMS search over the current OpenAI-backed
  index. Deterministic hash indexing remains available only for smoke/offline
  checks.
- Indexed corpus: 189 document records, including 24 metadata-only records.
- Local index artifacts: SQLite/FTS metadata store plus FAISS-compatible
  `IndexFlatIP` vector artifacts under `.data/qms-index`.
- Current vector contents: OpenAI `text-embedding-3-large` embeddings at 3072
  dimensions with normalized vectors and `embedding_provider=openai`.
- Hosted OpenAI File Search: synced local state for the current corpus hash with
  189 uploaded normalized Markdown files.
- SQLite store: documents/chunks/source files/ingest runs plus `revisions` and
  `doc_references`; latest status reports 2,355 references.
- Reranker: optional `sentence_transformers.CrossEncoder` implementation exists
  for `Qwen/Qwen3-Reranker-4B`; this local environment reports
  `deterministic_fallback` because `sentence_transformers` and the local model
  cache are unavailable.
- Answer synthesis: `gpt-5.5` synthesis is enabled by default through
  `ANSWER_SYNTHESIS_ENABLED=true`, uses only validated retrieved evidence, and
  falls back to deterministic extractive answers when the model call fails or
  returns unsupported citation labels.
- Current eval result: latest local OpenAI-index core eval run is 84/84 at
  threshold `0`, average score `0.5812`, recorded in
  `docs/eval-runs/2026-05-07-040446.md`. The quality gaps are explicit:
  citation validity `0.3773`, Recall@k `0.4504`, and obsolete leakage `0.3000`.
- Frontend: desktop-first Vite workbench with prompt composer, mode controls,
  citations, source drawer, inventory/debug tabs, model/index stats, and
  Playwright E2E coverage.
- Browser verification: Playwright MCP loaded the desktop app, ran an
  enumeration query, opened Sources and Debug, and reported zero console
  warnings/errors. Command-line Playwright is skipped by `scripts/check.sh` only
  for the known Codex macOS Chromium MachPort sandbox failure.

## Remaining Production Baseline

The remaining production baseline means:

- Keep OpenAI `text-embedding-3-large` document and query embeddings at 3072
  dimensions with provider parity enforced.
- Keep manifest/status fields that distinguish `embedding_provider=hash` from
  `embedding_provider=openai`, with production startup refusing hash vectors
  unless an explicit demo/degraded override is set.
- Keep hosted OpenAI File Search sync/retrieval healthy in search modes.
- Provision optional native dependencies and a local Qwen/BAAI model cache on
  any demo machine where real local reranking should run; otherwise keep the
  explicit deterministic fallback status in UI/evals.
- Keep `gpt-5.5` answer synthesis guarded by local citation validation and keep
  deterministic/extractive fallback available for degraded runs.
- Eval reports with retrieval, answer, citation, count, and abstention metrics.
- Final frontend source inspection, degraded status, and mode behavior verified
  through Playwright MCP/browser checks.

## Exact Gap List Against Original Plan

| Original area | Current status | What is left |
| --- | --- | --- |
| Repository setup | Mostly complete | Keep Playwright artifacts ignored, keep task state current after each milestone, and add final release/handoff notes after production gates pass. |
| Configuration | Mostly complete | Settings exist for the locked model baseline, chunking, retrieval, reranker, `QMS_USE_HASH_EMBEDDINGS`, provider status, and hosted state. Keep production validation strict against hash artifacts. |
| Corpus ingestion | Complete for MVP | Extraction handles DOCX, tables, Mac artifacts, metadata-only docs, manifests, normalized Markdown, and reference extraction. Remaining hardening: stronger section detection and richer data-quality warnings. |
| Local metadata store | Mostly complete | SQLite has documents/chunks/source files/ingest runs, FTS, `revisions`, and `doc_references`. Remaining work: richer exact lookup APIs, specialized revision/reference paths, and a stricter citation resolver that rejects unsupported source IDs. |
| Local FAISS/FTS retrieval | Complete for demo | Chunking, metadata chunks, FTS, vector search, rank fusion, OpenAI 3072-dimensional vectors, provider manifest, and production hash gates exist. Native `faiss.IndexFlatIP` is used when importable; otherwise status reports `numpy_fallback`. Remaining work: improve category-level retrieval quality. |
| Reranker | Complete with fallback | Optional `sentence_transformers.CrossEncoder` path exists for Qwen and preserves source metadata; deterministic fallback reports `warning=real_reranker_unavailable` when dependencies/model cache are missing. Remaining work: provision native deps/model cache and run reranker eval comparisons. |
| Hosted OpenAI File Search | Mostly complete | Hosted sync state is `synced` with 189 files for the current corpus hash and hosted retrieval is wired into modes. Remaining work: stronger hosted citation validation, failure-mode evals, and release-hardened fallback reporting. |
| Query planning | Mostly complete | Heuristic routing exists for known-item, enumeration, revision, cross-doc, compliance, extraction, and exploratory queries, including revision diff and multi-hop reference following. Remaining quality work: typo tolerance, stronger revision parsing, and optional `gpt-5.4-mini` fallback. |
| Answer generation and citations | Complete for demo | `gpt-5.5` synthesis over validated evidence is wired with deterministic fallback; answer citations remain local citation rows, not model-invented references. Remaining quality work: claim-level factuality judging and stronger answer evals. |
| API | Complete for demo | `/search`, `/chat`, `/stats`, `/status`, `/health`, `/documents`, and `/chunks` exist with structured response contracts. |
| Frontend | Complete for desktop demo | Workbench has prompt composer, source drawer, citations, modes, inventory, debug panel, fallback states, and desktop browser verification. |
| Eval dataset | Mostly complete | 84-case core dataset and harness exist with Recall@k-style, top-k, count, citation, latest revision, and obsolete leakage metrics. Remaining work: MRR/precision and LLM-judge answer scoring. |
| Eval reporting | Mostly complete | Markdown/JSON reports include commit SHA, corpus hash, index manifest hash, model config, retrieval mode, category metrics, and failures. Remaining work: previous-run deltas. |
| Ablations | Partial | Chunking tradeoffs are documented. Missing OpenAI vs hash, hosted vs local, reranker on/off, Qwen vs BAAI, top-K, and model/dimension ablations. |
| Documentation | Mostly complete | Design/readme/walkthrough/evals/indexing/troubleshooting docs exist and now reflect native fallback, synthesis, eval, and browser-verification status. Remaining work is final release/handoff notes after the last eval pass. |
| Commit/push/PR | Partial | Branch is pushed to the user fork and a draft PR exists there. Upstream PR creation is blocked by repository permissions. Need milestone commits for the remaining production work and final PR body update. |

## One-Shot Completion Sequence

The remaining work should be executed in this order. Each milestone has a red
test first, implementation, full test run, eval run, docs update, commit, and
push. Do not optimize for cost until the highest-quality baseline passes.

### 1. Production Config And Index Safety

Goal: prevent accidental production runs against degraded hash artifacts.

Tasks:

- Add `QMS_ENV=development|production` or equivalent runtime mode.
- Persist `embedding_provider` in `.data/qms-index/manifest.json` as `openai`
  or `hash`.
- Make production search startup fail if `embedding_provider=hash` unless an
  explicit degraded/demo flag is set.
- Add `sync-openai-file-search` script entry in `pyproject.toml`.
- Expose production/degraded state in `/stats` and the UI.

Tests:

- Config rejects production mode with hash embeddings.
- Stats shows `embedding_provider`.
- Existing hash-index E2E still works when `QMS_USE_HASH_EMBEDDINGS=true`.

Gate:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
cd frontend && npm run test:e2e
```

### 2. Real OpenAI Embedding Build

Status: complete for the current local artifact. Goal was to build the required
local FAISS-compatible index using `text-embedding-3-large` at 3072 dimensions.

Tasks:

- Run ingestion from the source zip without reading or printing `.env`.
- Run `uv run build-qms-index` without `--hash-embeddings`.
- Confirm OpenAI calls pass `dimensions=3072`.
- Confirm manifest records:
  - `embedding_model=text-embedding-3-large`
  - `embedding_dimensions=3072`
  - `embedding_provider=openai`
  - `faiss_index_type=IndexFlatIP`
  - `vectors_normalized=true`
  - corpus hash
- Commit large artifacts normally or through Git LFS if configured.

Commands:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run ingest-qms
UV_CACHE_DIR=/private/tmp/uv-cache uv run build-qms-index
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-status --tasks TASKS.md --index-dir .data/qms-index --openai-state .data/openai/vector_store_state.json
```

Tests:

- Built index dimension equals actual embedding length.
- Query embedding model/dimensions match document embedding model/dimensions.
- Search service refuses mismatched manifest/config.

Gate:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --mode local
```

### 3. Hosted OpenAI File Search

Status: synced for the current corpus hash. Goal was to make hosted retrieval a
real production path over normalized Markdown.

Tasks:

- Add `sync-openai-file-search` CLI.
- Compute corpus hash before upload.
- Reuse an existing vector store when the stored corpus hash matches.
- Upload only normalized Markdown files.
- Poll vector-store file processing to completion.
- Persist gitignored state in `.data/openai/vector_store_state.json`.
- Store file-ID to local document/revision/path mappings.
- Add hosted retrieval wrapper for `mode=hosted`.
- Add `mode=auto` behavior: hosted first when healthy, local fallback when not.
- Validate hosted citations against local metadata before answer generation.

Tests:

- Mocked OpenAI sync creates state without secrets.
- Repeated sync with same corpus hash reuses vector store.
- Hosted result file IDs resolve to local document/revision records.
- Hosted unavailable path falls back to local and reports a warning.

Gate:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run sync-openai-file-search
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-status --tasks TASKS.md --index-dir .data/qms-index --openai-state .data/openai/vector_store_state.json
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --mode hosted
```

No manual OpenAI UI setup is required. The agent can create the vector store,
upload files, poll status, and store the hosted IDs through the API using the
existing `OPENAI_API_KEY` in `.env`.

### 4. Real Reranker Integration

Status: implemented with environment-dependent activation. Goal was to replace
the placeholder with a real local backend when dependencies are available while
preserving explicit fallback behavior.

Tasks:

- Added optional `sentence_transformers.CrossEncoder` backend.
- Preserved Qwen `Qwen/Qwen3-Reranker-4B` as the default configured model.
- Preserved BAAI `BAAI/bge-reranker-v2-m3` as the documented constrained-dev
  fallback model config.
- Preserved chunk IDs, source IDs, and citation metadata through rerank.
- Added degraded-mode warnings if the configured reranker cannot load.
- Recorded reranker state in `/stats`, eval reports, and UI debug data.

Tests:

- Reranker receives top-N fused candidates.
- Reranker returns top-K evidence with citation IDs intact.
- Failed Qwen dependency/model load falls back to deterministic ranking with
  warning.
- Reranker-disabled mode preserves pre-rerank order.

Gate:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --mode local
```

### 5. Revision, Reference, And Multi-Hop Retrieval

Status: storage tables are present; specialized query paths remain. Goal: handle
the hardest interview categories without relying on top-N luck.

Tasks:

- Keep `revisions` table keyed by canonical document family.
- Keep `doc_references` table extracted from normalized text and filenames.
- Implement latest-active revision lookup as the default.
- Implement explicit historical revision lookup.
- Implement obsolete/signed filters.
- Implement revision diff path for queries like `Rev C vs Rev D`.
- Implement reference-following path for traceability questions.
- For enumeration queries, force SQL inventory/list behavior.

Tests:

- BOM latest defaults to Rev G where applicable.
- Specific historical revisions return the requested revision.
- Obsolete docs are excluded unless requested.
- Revision diff cites both revisions.
- Traceability answers include at least two linked source documents.
- Counts are exact SQL counts with listed backing records.

Gate:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --mode hybrid
```

### 6. GPT-5.5 Answer Synthesis With Strict Citations

Goal: produce useful answers while preventing unsupported claims.

Tasks:

- Build evidence packets from validated local/hosted retrieval results.
- Generate answers with `CHAT_MODEL=gpt-5.5`.
- Require citation markers for factual claims.
- Add claim-to-source validator before returning final output.
- Add refusal behavior when evidence is missing or contradictory.
- Support answer shapes:
  - exact document result
  - grouped inventory
  - table/list extraction
  - revision diff
  - traceability chain
  - compliance gap analysis
- Keep the raw answer and structured citations aligned.

Tests:

- Known-item answer cites the latest source.
- Enumeration answer count equals SQL count.
- Unsupported claim is refused or marked uncertain.
- Invalid citation IDs are rejected.
- Cross-document answer includes source chain citations.

Gate:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --mode auto
```

### 7. Eval Upgrade And Quality Loop

Goal: make evals useful enough to guide the remaining search improvements.

Tasks:

- Add expected document/revision targets to the 84 core cases.
- Compute Recall@5, Recall@10, Recall@20, MRR, and Precision@k.
- Add exact-count scoring for enumeration cases.
- Add citation precision/coverage scoring.
- Add unanswerable false-premise abstention checks.
- Add scoped `gpt-5.5` binary judge only where deterministic checks are not
  enough.
- Write raw traces to `.data/eval-runs/<timestamp>/traces.jsonl`.
- Write Markdown reports with commit SHA, corpus hash, manifest hash, model
  config, retrieval mode, metrics, failures, and next fixes.

Quality loop:

1. Run all 84 core evals.
2. Classify failures by retrieval, reranking, query planning, answer synthesis,
   citation, or eval-data issue.
3. Fix the highest-impact class.
4. Record the fix in `docs/eval-runs/` and `docs/bugs/bugs.md` when applicable.
5. Repeat until category-level pass rates are acceptable for demo.

### 8. Frontend Production Pass

Goal: make the UI inspectable enough for the live walkthrough.

Tasks:

- Add clickable source inspector or drawer.
- Add source path, doc ID, revision, section, chunk, and metadata status views.
- Add hosted/local/hybrid/degraded banners.
- Add latest/all revisions and include-obsolete controls.
- Add debug drawer for query plan, retrieval scores, reranker state, and stats.
- Keep every long filename/document ID wrapped on mobile.

Tests:

- Local E2E checks desktop and mobile load during development.
- Search renders citations and source inspector.
- Stats panel shows model/index/reranker/hosted status.
- No horizontal overflow on mobile after result rendering.
- No console errors during the search flow.

Gate:

```bash
cd frontend
npm run build
npm run test:e2e
```

Playwright MCP/browser verification has been run against the desktop workbench
after the production UI/API behavior stabilized. Command-line Playwright remains
part of `scripts/check.sh`; only the known Codex macOS Chromium MachPort failure
is treated as a sandbox skip.

### 9. Documentation, PR, And Handoff

Goal: make the review story coherent and auditable.

Tasks:

- Update `DESIGN.md` with the final architecture actually shipped.
- Update README Andrew notes with real commands and actual default URLs.
- Keep `docs/indexing.md` current after reranker, answer synthesis, and eval
  upgrades land.
- Update `docs/evals.md` with final scoring methodology and latest pass rates.
- Update `docs/walkthrough.md` with demo script and expected outputs.
- Update `docs/troubleshooting.md` with known startup, hosted sync, reranker, and
  Playwright failure modes.
- Update draft PR body with:
  - implementation summary
  - model/index settings
  - eval results
  - known limitations
  - demo instructions

Final validation:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-status --tasks TASKS.md --index-dir .data/qms-index --openai-state .data/openai/vector_store_state.json
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --mode auto
cd frontend && npm run build && npm run test:e2e
```

After those commands pass in a normal browser-capable environment, run the
Playwright command-line E2E suite without the sandbox skip to validate browser
assertions end to end.

## Production Acceptance Gates

The app should not be considered production-ready until all of these are true:

- Local index is built with OpenAI `text-embedding-3-large` 3072 embeddings.
- Manifest records provider, model, dimensions, FAISS type, normalization, corpus
  hash, and chunking settings.
- Hosted OpenAI File Search sync succeeds and hosted state exists locally.
- Hosted, local, hybrid, and auto modes behave distinctly and report fallback.
- Qwen reranker or documented fallback is active and visible in stats.
- `/chat` and `/search` return citation-grounded answers.
- Revision and enumeration queries use deterministic SQL/revision logic.
- Cross-document queries follow extracted references.
- Every factual answer has valid citations.
- Full 84-case eval suite runs with category-level metrics and Markdown report.
- Frontend passes build and desktop browser verification; command-line
  Playwright passes in normal environments or records the known Codex macOS
  MachPort sandbox skip.
- Docs reflect actual behavior, not intended behavior.
- Latest milestone commit is pushed to the draft PR branch.
