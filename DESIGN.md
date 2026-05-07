# MedAI QMS Search Design

## Objective

Build an internal-search demo for a fictional MedAI Quality Management System
(QMS) corpus for the MX1 portable X-ray system. The product experience is a chat
assistant that answers questions from customer-owned documents with strict
citations.

This implementation is scoped to customer document search only. Regulatory
search, web search, SharePoint/Drive connectors, authentication, and production
multi-tenant controls are out of scope for the MVP.

## Corpus

- Source artifact: `Example_QMS_-_MedAI.zip`
- Expected content: 189 real `.docx` files after ignoring `__MACOSX` and resource
  fork entries.
- Observed document families: `BOM`, `DHF`, `DMR`, `ECR`, `ESF`, `IFU`, `MEMO`,
  `PLN`, `QSR`, `RSK`, `TRA`, `VVAM`, `VVPR`, and related prefixes.
- Important metadata is encoded in filenames: document ID, revision, signed
  status, obsolete status, product/version references, and workstation or system
  identifiers.

## Query Patterns

The MVP must support:

- Known-item retrieval by document ID, title terms, revision, or filename.
- Exploratory search across QMS topics.
- Compliance cross-reference across plans, reports, risk files, and trace
  matrices.
- Content extraction and short synthesis from retrieved passages.
- Revision and change tracking, with latest active revision preferred.
- Cross-document analysis when multiple records are required.
- Enumeration and counting, with transparent caveats when counts depend on
  metadata-only documents or incomplete extraction.

## Target Production Baseline Defaults

Use these defaults for the remaining production baseline unless there is a
documented blocker:

| Concern | Baseline |
| --- | --- |
| Answer model | `gpt-5.5` |
| Embedding model | `text-embedding-3-large` |
| Embedding dimensions | `3072` |
| Vector index | FAISS `IndexFlatIP` |
| Vector scoring | Inner product over L2-normalized vectors |
| Reranker | Qwen reranker, highest-quality local or hosted variant available |
| Retrieval mode | Hybrid lexical plus dense retrieval, followed by reranking |
| Citation policy | No sourced claim without a citation |

Implementation notes:

- Configure OpenAI embeddings with `dimensions=3072`; reject existing indexes with
  a different dimension.
- Normalize vectors before adding them to `IndexFlatIP`, so inner product behaves
  like cosine similarity.
- The Qwen reranker should rerank the merged candidate set rather than every
  corpus chunk. If it is unavailable, continue with a documented fallback and mark
  the eval run as degraded.

## Architecture

Planned runtime flow:

```text
QMS zip
  -> binary .docx extraction
  -> metadata normalization
  -> chunking
  -> embeddings + lexical index
  -> persisted local index
  -> hybrid retrieval
  -> Qwen rerank
  -> answer synthesis with citations
  -> CLI and fullstack chat
```

Recommended module boundaries:

- `src/agent/search/corpus.py`: zip scanning, `.docx` extraction, corpus records.
- `src/agent/search/metadata.py`: filename parsing, revision ordering, active vs
  obsolete status, document family normalization.
- `src/agent/search/chunking.py`: section-aware chunks and citation spans.
- `src/agent/search/indexing.py`: embedding generation, FAISS persistence,
  lexical index persistence, index manifest.
- `src/agent/search/retrieval.py`: hybrid retrieval, candidate merging,
  deduplication, reranking.
- `src/agent/search/answering.py`: prompt construction, strict citation handling,
  refusal/uncertainty policy.
- `src/agent/search/eval_types.py`: shared schemas for eval cases and results.
- `src/agent/server.py`: add search-aware chat and health/index endpoints.
- `src/agent/cli.py`: add search mode and index status commands.

The exact file layout can change if another worker finds a cleaner fit, but the
boundaries should remain independently testable.

## Index Manifest

Every built index should include a manifest with:

- Corpus artifact path and SHA-256.
- Build timestamp and git commit.
- Document count, skipped count, chunk count, and metadata-only count.
- Embedding model and dimensions.
- FAISS index type.
- Lexical index implementation.
- Reranker name and version/config.
- Chunking settings.
- Known extraction warnings.

The app should refuse to serve a stale or incompatible index unless explicitly
started in a rebuild or degraded mode.

## Retrieval Contract

Retrieval returns structured evidence, not plain text blobs:

- `doc_id`
- `title`
- `filename`
- `revision`
- `status` (`active`, `obsolete`, `unknown`)
- `signed` boolean or `unknown`
- `chunk_id`
- `heading` or nearest section label
- `content`
- `scores` for lexical, dense, fused, and rerank where available

Ranking rules:

- Exact document ID and filename matches get a strong boost.
- Latest active revision wins by default when multiple revisions match.
- Obsolete documents can be used only when directly requested or when needed to
  explain revision history.
- Sparse, metadata-only documents may be returned for known-item queries, but
  should not be used for unsupported synthesis.

## Answer Contract

Answers must:

- Cite every factual claim that comes from the corpus.
- Prefer concise synthesis over dumping long passages.
- Say when evidence is missing, contradictory, obsolete, or metadata-only.
- Distinguish current active records from obsolete records.
- Include a compact source list with filename, revision/status, and chunk/section.
- Avoid using model knowledge for QMS facts not present in retrieved evidence.

## API Shape

Target endpoints:

- `GET /health`: app health and configured model names.
- `GET /index/status`: manifest summary, ready/degraded state, document/chunk
  counts, and build warnings.
- `POST /search`: query plus optional filters; returns ranked evidence.
- `POST /chat`: existing chat contract extended to use retrieval and return
  answer citations.

List responses should always be guarded with `Array.isArray` or equivalent on the
client side. Mutation and indexing handlers should return structured errors.

## Frontend Direction

The fullstack MVP should be a search workbench, not a marketing page:

- Primary chat surface with citations visible inline.
- Source drawer or side panel for retrieved documents.
- Index status and degraded-mode indicators.
- Search mode controls for latest-active vs include-obsolete behavior.
- Loading, empty, error, and partial-success states.
- Keyboard navigable controls and accessible names for icon-only actions.
- Desktop-first layout for the live onsite walkthrough, with mobile containment
  kept as a regression requirement rather than the primary information density
  target.

Use the existing React/Vite app unless implementation work proves the stack is
fighting the workflow.

## Evaluation Strategy

Use test-driven development for search:

1. Add failing fixture-backed tests for metadata parsing, extraction, chunking,
   and retrieval.
2. Implement the smallest indexer/retriever that passes deterministic tests.
3. Add retrieval evals over fixed QMS queries before adding answer synthesis.
4. Add answer evals only after retrieval quality is measurable.
5. Record every material run in `docs/eval-runs/`.

Core metrics:

- Recall@k for expected source documents.
- MRR for known-item and exact-ID queries.
- Citation precision and citation coverage.
- Answer correctness against expected facts.
- Abstention quality when evidence is missing.
- Revision handling accuracy.

## Tradeoffs

| Decision | Why | Cost |
| --- | --- | --- |
| Local FAISS over managed vector DB | Fast, portable demo with no service setup | No production ACLs, replication, or managed scaling |
| `IndexFlatIP` over IVF/HNSW | Exact search is simple and high quality for a small corpus | Linear scan will not scale to large corpora |
| Hybrid retrieval over dense-only | Better exact IDs, filenames, revisions, and acronyms | More indexing and scoring code |
| Qwen reranker over no rerank | Improves final evidence ordering for synthesis | Adds latency and operational dependency |
| Filename metadata first | QMS revision/status signals are encoded in filenames | Must handle inconsistent naming carefully |
| Strict citations | Builds trust and makes evals concrete | Forces abstention when extraction is weak |
| `.docx` body extraction first | Matches corpus format and avoids brittle binary parsing | Page-level citations may be unavailable |

## Risks And Mitigations

- Empty or low-text `.docx` files: mark as metadata-only, return for known-item
  queries, and avoid unsupported synthesis.
- Revision ambiguity: centralize revision parsing and add table-driven tests.
- Dimension mismatch: enforce `3072` in manifest and fail fast on load.
- Reranker outages: fallback to fused hybrid score, surface degraded status, and
  tag eval runs.
- Hallucinated citations: answer builder should only cite retrieved evidence
  objects and tests should check citation IDs.
- Counting questions: count over normalized document records first, then explain
  whether the count covers documents, chunks, or extracted mentions.

## Status Commands

Current repo/status commands:

```bash
git branch --show-current
git status --short
git diff --stat
rg --files DESIGN.md TASKS.md docs
uv run pytest evals/ -v
```

Implemented search commands:

```bash
uv run ingest-qms
uv run build-qms-index
uv run build-qms-index --hash-embeddings
uv run search-status --tasks TASKS.md --index-dir .data/qms-index --openai-state .data/openai/vector_store_state.json
uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --hash-embeddings --mode local
uv run serve
cd frontend && npm run dev
```

## Current Implementation Status

- Ingestion keeps every real DOCX represented in the index. Sparse or empty-body
  documents get metadata chunks so known-item retrieval can still find them.
- The local index manifest records `text-embedding-3-large`, 3072 dimensions,
  normalized vectors, and FAISS `IndexFlatIP`.
- Chunking uses 600-token child chunks with 100-token overlap, metadata chunks,
  row-preserving table chunks, and answer-time neighbor expansion.
- Current vector artifacts are the OpenAI local index:
  `embedding_provider=openai`, `text-embedding-3-large`, 3072 dimensions,
  normalized vectors, and `7,778` chunks.
- SQLite includes document, chunk, source-file, ingest-run, revision, and
  `doc_references` tables; current status reports `2,355` references.
- Hosted OpenAI File Search state is synced for the current corpus hash with
  `189` normalized Markdown files.
- `build-qms-index --hash-embeddings` remains the deterministic local smoke path.
  `build-qms-index` is the OpenAI build path and is the current indexed
  baseline.
- The configured Qwen reranker is not yet the active backend; status currently
  reports `deterministic_fallback`.
- The latest committed 84-case local hash eval is `25 / 84`, average `0.5210`,
  in `docs/eval-runs/2026-05-07-014026.md`. It is a baseline failure report,
  not a finished-quality claim.
- Playwright MCP/browser verification is deferred to final verification after
  production search behavior and the frontend production pass are complete.

See `docs/architecture-decisions.md` for the running trade-off log and chunking
decision evidence.
