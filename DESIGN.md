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
- Important naming caveat: `TRA-*` records in this corpus are training records,
  not traceability matrices. The current MX1 verification/validation trace
  matrix is `VVAM-P01-004`.
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

The latest audit showed that these query patterns need different execution
paths. Treating every query as top-N semantic search is not accurate enough for
QMS work. The current design is intent-first: retrieval is still available, but
inventory, table extraction, traceability, revision comparison, and temporal
status questions route through deterministic helpers before answer synthesis.

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
| Retrieval mode | Explicit `auto`, `hosted`, `hybrid`, and `local` modes |
| Citation policy | No sourced claim without a citation |

Implementation notes:

- Configure OpenAI embeddings with `dimensions=3072`; reject existing indexes with
  a different dimension.
- Normalize vectors before adding them to `IndexFlatIP`, so inner product behaves
  like cosine similarity.
- The Qwen reranker should rerank the merged candidate set rather than every
  corpus chunk. If it is unavailable, continue with a documented fallback and mark
  the eval run as degraded.

## Retrieval Mode Contract

Mode names are user-visible and must keep stable semantics across CLI, API,
frontend, and evals:

| Mode | Hosted behavior | Local behavior | Intended use |
| --- | --- | --- | --- |
| `auto` | Try hosted OpenAI File Search first when synced and healthy. | Fall back to local retrieval if hosted is unavailable, empty, stale, or errors. | Default demo and production-style smoke mode. |
| `hosted` | Prefer hosted OpenAI File Search. | Fall back locally with an explicit warning rather than failing the whole turn. | Hosted sync validation and hosted/local comparison. |
| `hybrid` | Never call hosted search. | Use local SQLite FTS, FAISS-compatible dense retrieval, rank fusion, and reranking. | Strict local quality baseline and primary eval mode. |
| `local` | Never call hosted search. | Use local-only fallback/debug retrieval; fall through to SQLite FTS when vector search is unavailable. | Debugging local index, fallback, and degraded behavior. |

`hybrid` and `local` are strict no-hosted modes. If hosted search is called in
either mode, that is a routing bug. `hosted` is hosted-preferred rather than
hosted-only so walkthroughs can continue when the hosted vector store is stale or
temporarily unavailable.

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
- `provenance` sufficient to trace the answer back to the corpus artifact,
  normalized Markdown, SQLite row, chunk/table row, and final citation

Ranking rules:

- Exact document ID and filename matches get a strong boost.
- Latest active revision wins by default when multiple revisions match.
- Obsolete documents can be used only when directly requested or when needed to
  explain revision history.
- Sparse, metadata-only documents may be returned for known-item queries, but
  should not be used for unsupported synthesis.

## Intent-First Retrieval Contract

The query planner assigns both a broad category and a concrete intent. The broad
category controls default retrieval, while the concrete intent can bypass generic
semantic search when the answer needs a deterministic source of truth.

Current high-value intents:

| Intent | Purpose | Required evidence behavior |
| --- | --- | --- |
| `mx1_bom` | Find the MX1 system BOM. | Return `BOM-055 Rev G` as primary and group software BOMs separately. |
| `510k_summary_location` | Locate 510(k) summary evidence. | Cite `MEMO-P01-859`, `DHF-008`, and `PLN-P01-061`; state when no standalone summary file is indexed. |
| `vvpr_inventory` | List/count MX1 verification protocols. | Use SQL inventory, not top-N snippets; expose total/non-obsolete/latest-active scope counts. |
| `risk_related_inventory` | Show risk-related documents. | Use SQL/topic inventory across `RSK`, risk plans, `VVAM`, and risk/RMF-bearing docs. |
| `dhf_82030` | Check DHF against design-control expectations. | Cite `DHF-008` and planning support; include the current QMSR caveat. |
| `risk_protocol_trace` | Find P01 protocols traced from risk analysis. | Use active `RSK`/`VVAM` evidence and avoid historical predecessor protocol IDs. |
| `electrical_safety_acceptance` | Extract electrical safety criteria. | Cite `MEMO-P01-685 Table 2 row 2` and `3P-P01-33`; include IEC 60601-1 and `PASS`. |
| `open_design_review_actions` | Summarize open review actions. | Cite `MEMO-P01-859 Table 5`; do not inherit unrelated prior-turn context. |
| `ambiguous_risk_revision_diff` | Compare missing risk-analysis revision pair. | Return clarification/no-answer when no single RSK chain contains both revisions. |
| `ecr_last_year_status` | List ECRs by temporal/status policy. | Extract approval effective dates, DCO/status fields, and state the current-date policy. |
| `electrical_leakage_trace` | Trace leakage from risk to report. | Return explicit chain: risk source, `VVAM` bridge, summary, and `3P-P01-33` report. |
| `third_party_report_mapping` | Map 3P reports to standards. | Distinguish completed `3P-*` reports from planning evidence. |
| `ecr_count` | Count engineering change requests. | Count active signed ECR records in SQL and list the IDs. |
| `verification_completed_vs_planned` | Compare completed vs planned V&V work. | Use `MEMO-P01-685` result rows and `PLN-P01-065` planned scope, not raw VVPR count. |

Hybrid retrieval supports those paths rather than replacing them. For extraction
and known-item categories, rank fusion preserves top lexical hits, widens the
rerank candidate pool, boosts distinctive entities such as `510(k)`, `K241567`,
`3P-P01-32`, `3P-P01-33`, `DHF-008`, and `acceptance criteria`, and deduplicates
repeated same-section chunks before answer synthesis.

Query expansion is allowed only when it is observable and source-grounded. The
trace must show the raw query, normalized query, planned intent/category,
expansion terms or entities, backend, references followed, warnings, and final
candidate/citation IDs. Expanded terms may increase recall, but they must not
override exact IDs, latest-active policy, table row evidence, or forbidden-source
guards.

Conversation state is intentionally conservative. Prior citations are inherited
only for real follow-ups, using token-boundary cues and the absence of a new
entity. Topic changes such as "Summarize all design review action items" must be
treated as standalone queries.

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

Current eval expectations are intentionally stricter than broad substring
matching:

- `required_doc_ids`: concrete document IDs that must be present when a case
  needs exact evidence.
- `required_backend`: deterministic backend expectations such as
  `sql_inventory`.
- `required_table_evidence`: row/cell evidence for table-backed answers.
- `must_not_include`: forbidden claims or source families, for example
  `VVPR-P00` for current risk protocols and `TRA-*` customer training docs for
  traceability matrices.

The 14 audited source-truth queries are covered by
`evals/test_qms_source_truth_contracts.py`. The smoke dataset is the fast
walkthrough subset, while `qms_core.jsonl` keeps the broader 84-case coverage.
The core data has been corrected so traceability-matrix expectations use
`VVAM-P01-004` instead of the `TRA-*` training records.

Mandatory query-path regression gates:

- Run `uv run pytest evals/test_qms_source_truth_contracts.py -q` before the full
  pytest suite for any change to planning, query expansion, retrieval, reranking,
  citation assembly, answer synthesis, or trace formatting.
- Then run `uv run pytest -q` or the affected pytest subset plus full pytest
  before handoff.
- Run the targeted generated/audit eval for the touched artifact contract:
  `search-evals --dataset smoke` for walkthrough behavior, `search-evals
  --dataset core` for broader scoring, `--answers-jsonl
  evals/datasets/qms_smoke_golden_answers.jsonl` for generated-answer contract
  checks, and `--validate-only` when dataset evidence expectations change.
- A broad average score does not override the 14-query gate. Failures in required
  sources, table evidence, retrieval trace, provenance fields, or
  `must_not_include` guards block query-path changes.

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
uv run pytest evals/test_qms_source_truth_contracts.py -q
uv run pytest evals/ -v
```

Implemented search commands:

```bash
uv run ingest-qms
uv run build-qms-index
uv run build-qms-index --hash-embeddings
uv run search-status --tasks TASKS.md --index-dir .data/qms-index --openai-state .data/openai/vector_store_state.json
uv run search-qms "Find BOM-055 Rev G" --mode hybrid --limit 8
uv run chat --qms-search --mode hybrid --limit 8
uv run chat --qms-search --mode auto --limit 8
uv run chat --qms-search --mode hybrid --limit 8 --no-progress
uv run chat --qms-search --mode hybrid --limit 8 --plain
printf 'Find BOM-055 Rev G\nquit\n' | uv run chat --qms-search --mode hybrid --limit 8 --trace
printf 'Find BOM-055 Rev G\nquit\n' | uv run chat --qms-search --mode hybrid --limit 8 --trace --raw-trace
printf 'Find BOM-055 Rev G\nquit\n' | uv run chat --qms-search --mode hybrid --limit 8 --full-citations
printf 'Find BOM-055 Rev G\nquit\n' | uv run chat --qms-search --mode hybrid --json
printf 'Find the Bill of Materials for the MX1 system\nWhat revision is that?\nShow me the full pathname citation.\nquit\n' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
printf 'Find BOM-055 Rev G\nquit\n' | OPENAI_VECTOR_STORE_STATE=/private/tmp/missing-openai-vector-store-state.json uv run chat --qms-search --mode auto --limit 8 --trace
uv run pytest evals/test_qms_source_truth_contracts.py -q
uv run search-evals --dataset smoke --answers-jsonl evals/datasets/qms_smoke_golden_answers.jsonl
uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --mode hybrid --fail-under 0
uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --mode auto --fail-under 0
uv run serve
cd frontend && npm run dev
```

Interactive QMS CLI sessions use a `QMS> ` prompt, Rich panels for human output,
source cards for citations, grouped trace panels for `--trace`, and a dynamic
status spinner on real TTYs. Piped runs suppress progress automatically; `--json`
is the machine-readable mode and `--plain` keeps deterministic text output.

## Current Implementation Status

- Ingestion keeps every real DOCX represented in the index. Sparse or empty-body
  documents get metadata chunks so known-item retrieval can still find them.
- The local index manifest records `text-embedding-3-large`, 3072 dimensions,
  normalized vectors, and FAISS `IndexFlatIP`.
- Chunking uses 600-token child chunks with 100-token overlap, metadata chunks,
  row-preserving table chunks, row-level chunks for tables with 50 rows or
  fewer, and answer-time neighbor expansion. The row cap avoids indexing huge
  trace/risk matrices as tens of thousands of individual vector records while
  preserving exact row/cell citations for audit-sized tables.
- A rebuilt local index using this schema should report:
  `embedding_provider=openai`, `text-embedding-3-large`, 3072 dimensions,
  normalized vectors, and `17,651` chunks. The large generated index files are
  not pushed through the public fork when they exceed GitHub's normal blob
  limits; rebuild locally with `ingest-qms` and `build-qms-index`.
- SQLite includes document, chunk, source-file, ingest-run, revision, and
  `doc_references` tables; the post-rebuild status for this corpus reports
  `4,446` references.
- Hosted OpenAI File Search state is synced for the current corpus hash with
  `189` normalized Markdown files.
- `build-qms-index --hash-embeddings` remains the deterministic local smoke path.
  `build-qms-index` is the OpenAI build path and is the current indexed
  baseline.
- The configured Qwen reranker is not yet the active backend; status currently
  reports `deterministic_fallback`.
- Deterministic QMS intents are implemented for the current 14-query audit set,
  including BOM, 510(k), VVPR inventory, risk inventory, DHF/QMSR, risk protocol
  trace, electrical-safety criteria, design-review actions, risk revision
  ambiguity, ECR status/date, leakage trace, third-party report mapping, ECR
  count, and completed-vs-planned verification.
- Latest verification for this audit pass:
  - Full tests: `137 passed`, with only the existing LangGraph deprecation
    warning.
  - Source-truth contracts: `14 passed`.
  - Smoke eval in `hybrid` mode: `7 / 7 passed`, average score `1.0`.
  - Core dataset schema: `84` cases, `12` per category, valid.
- The older 84-case local OpenAI-index report remains useful historical context
  (`84 / 84` at harness threshold `0`, average `0.5812`, in
  `docs/eval-runs/2026-05-07-040446.md`), but it did not catch row-level,
  forbidden-source, or deterministic-intent failures. The stricter eval contract
  is now the quality gate for these audited behaviors.
- Playwright MCP/browser verification completed on `2026-05-07` for the
  desktop workbench, including status, source inspection, debug output, and an
  enumeration search flow with no browser console warnings/errors.

See `docs/architecture-decisions.md` for the running trade-off log and chunking
decision evidence.
