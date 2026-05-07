# MedAI QMS Search Tasks

This file tracks the implementation plan for the MedAI QMS internal-search MVP.
Worker A owns documentation and task scaffolding only.

## Status Commands

Run these before starting or handing off work:

```bash
git branch --show-current
git status --short
git diff --stat
rg --files DESIGN.md TASKS.md docs
uv run pytest evals/ -v
```

Run these after frontend changes:

```bash
cd frontend
npm install
npm run build
```

Implemented search commands:

```bash
uv run ingest-qms
uv run build-qms-index
uv run build-qms-index --hash-embeddings
uv run search-status --tasks TASKS.md --index-dir .data/qms-index --openai-state .data/openai/vector_store_state.json
uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --hash-embeddings --mode local
```

## Phase 0 - Planning And Guardrails

- [x] Add root design document with architecture, defaults, tradeoffs, and risks.
- [x] Add task tracker with implementation phases and status commands.
- [x] Add walkthrough, indexing, eval, troubleshooting, and bug docs.
- [x] Add `docs/eval-runs/.gitkeep` so eval run notes have a tracked home.
- [ ] Confirm with implementers before changing files outside docs/task tracking.

## Phase 1 - Corpus Ingestion

- [x] Add deterministic tests for zip scanning that ignore `__MACOSX` and resource
  fork entries.
- [x] Add binary `.docx` extraction; do not use `.text()` or text-based parsers on
  `.docx` files.
- [x] Emit document records with filename, extracted text, tables when available,
  warnings, and extraction status.
- [x] Preserve sparse/empty-body documents as metadata-only records so known-item
  retrieval can still find them by filename, ID, revision, and status.
- [x] Compute corpus SHA-256 and document-level hashes for manifests.

Dependencies: none beyond local corpus artifact and Python dependencies selected
by the implementer.

## Phase 2 - Metadata Normalization

- [x] Add table-driven tests for document IDs, prefixes, revisions, signed status,
  obsolete status, and product/version tokens.
- [x] Parse document IDs such as `VVPR-P01-179`, `BOM-055`, `MEMO-P01-638`, and
  `VVAM-P01-004`.
- [x] Normalize revision labels from title suffixes such as `_A`, `_B-signed`,
  `_C-Obsolete`, and `Rev G`.
- [x] Implement latest-active revision selection with obsolete records excluded by
  default.
- [x] Preserve source filename exactly for citations and auditability.

Dependencies: Phase 1 document records.

## Phase 3 - Chunking And Index Build

- [x] Add tests for chunk IDs, metadata carry-through, and index dimensionality.
  re-indexing.
- [x] Chunk by heading/section where possible, using 600-token child chunks,
  100-token overlap, metadata chunks, row-preserving table chunks, and
  answer-time neighbor expansion.
- [x] Embed chunks with `text-embedding-3-large` at `3072` dimensions.
- [x] Normalize vectors and store them in a FAISS-compatible `IndexFlatIP` local store.
- [x] Build a lexical index for exact IDs, filenames, acronym-heavy queries, and
  revision terms.
- [x] Persist a manifest containing corpus hash, model names, dimensions, counts,
  warnings, and build settings.

Dependencies: Phases 1 and 2.

## Phase 4 - Retrieval And Reranking

- [ ] Add retrieval eval fixtures for known-item, exploratory, cross-reference,
  revision, extraction, and counting queries.
- [ ] Merge lexical and dense candidates with deduplication by document/chunk.
- [ ] Boost exact document ID, filename, latest-active revision, and signed active
  records where appropriate.
- [ ] Rerank the merged candidate set with the Qwen reranker.
- [ ] Provide a degraded fallback when reranking is unavailable and report it in
  status/eval output.
- [ ] Return structured evidence objects with scores and citation metadata.

Dependencies: Phase 3 index.

## Phase 5 - Answer Synthesis

- [ ] Add answer tests that verify groundedness, citation coverage, and abstention.
- [ ] Use `gpt-5.5` as the default answer model.
- [ ] Build prompts from structured evidence only.
- [ ] Require citations for sourced claims and block invented citation IDs.
- [ ] Distinguish active, obsolete, metadata-only, and conflicting evidence.
- [ ] Add concise source summaries for every answer.

Dependencies: Phase 4 retrieval.

## Phase 6 - CLI And API

- [x] Add index build/status commands without breaking the existing `chat` command.
- [x] Add `POST /search` for ranked evidence.
- [ ] Extend `POST /chat` to return answer text plus citations.
- [x] Add `GET /stats` for manifest and degraded-mode details.
- [x] Wrap mutation/indexing handlers in `try/catch` or Python equivalent and
  return structured errors.
- [x] Guard list and object responses on the frontend before rendering.

Dependencies: Phases 3 to 5.

## Phase 7 - Frontend Workbench

- [x] Replace the bare chat UI with a dense search workbench.
- [x] Show answer citations in a source inspection panel.
- [x] Add index readiness, loading, empty, error, and
  partial-success states.
- [x] Provide retrieval mode controls for auto, hosted, local, and hybrid.
- [x] Verify visible focus rings and accessible labels in component markup.
- [x] Run frontend build after UI changes.

Dependencies: Phase 6 API.

## Phase 8 - Evals And Release Evidence

- [x] Add deterministic unit tests for parsing, chunking, indexing, and retrieval.
- [x] Add local retrieval and answer eval scaffold.
- [x] Record eval runs in `docs/eval-runs/YYYY-MM-DD-<summary>.md`.
- [ ] Include commit SHA, corpus hash, index manifest, model defaults, metrics, and
  notable failures in each run note.
- [x] Run `uv run pytest evals/ -v` before handoff.
- [x] Run `npm run build` after frontend work.

Dependencies: all implementation phases.

## Bug Triage Policy

- If a failure happens more than twice in a row, stop retrying the same fix path.
- Before the third attempt, check `docs/bugs/known_failures.md`.
- If the bug is not documented, add an entry with the failure signature, command,
  context, attempted fixes, why they failed, and next hypotheses.
- Once resolved, update the same entry with the confirmed fix and verification
  steps.
- Do not reapply a failed fix unless new evidence justifies it.
