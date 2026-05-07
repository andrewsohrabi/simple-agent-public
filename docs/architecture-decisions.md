# Architecture Decisions And Trade-Off Log

This file records implementation trade-offs, eval results, and decision context
for the MedAI QMS search assistant. It is intentionally more operational than
`DESIGN.md`: when we make a practical decision during development, record the
evidence here so future fixes do not have to rediscover the same context.

## 2026-05-07 - Highest-Quality Model Baseline

Decision:

- Start with `text-embedding-3-large` at full `3072` dimensions.
- Use FAISS `IndexFlatIP` with normalized vectors.
- Use `gpt-5.5` for chat, agent behavior, enrichment, and eval grading.
- Use deterministic query planning first; reserve `gpt-5.4-mini` for fallback
  query rewrite/classification.
- Keep Qwen reranker defaults: `Qwen/Qwen3-Reranker-4B`, top-N `80`, top-K `16`.

Reasoning:

- Accuracy, citation quality, and retrieval quality matter more than cost for
  the first onsite baseline.
- The corpus is small enough that upfront embedding cost is acceptable.
- Lower dimensions and cheaper models are ablation experiments after quality
  gates improve, not the initial architecture.

Verification:

- Config tests assert the defaults.
- `/stats` exposes model and index configuration.
- Local index manifest records embedding model, dimensions, vector index type,
  and normalization.

## 2026-05-07 - Local Review Index Artifacts

Decision:

- Commit the local review corpus/index artifacts for portability.
- Keep hosted OpenAI vector-store state and raw trace-style artifacts local-only.
- Store the committed local artifacts as normal Git blobs for the fork push,
  because GitHub rejected new LFS object uploads to the public fork.

Reasoning:

- Reviewers should be able to inspect and run the local fallback path without
  rebuilding everything.
- The largest current artifact, `.data/qms-index/qms.sqlite`, is just under
  GitHub's 100 MiB per-file hard limit. The vector file and metadata file are
  also under that limit.
- The chunking/indexing decisions are not constrained by artifact size; if a
  future remote allows LFS, these same artifacts can be moved back to LFS.

Current artifact state:

- Corpus zip: about 30 MB.
- Local index directory after the token-aware rebuild: about 278 MB.
- `.data/qms-index/qms.sqlite`: 104,816,640 bytes.
- `.data/qms-index/vectors.npy`: 95,576,192 bytes.
- `.data/qms-index/vector_metadata.json`: 56,602,752 bytes.

## 2026-05-07 - Initial Character-Chunk Ablation

Context:

- Before the final token-aware chunking instruction, we tested character-based
  chunk sizes using the deterministic hash embedding smoke path.
- This was useful as a directional signal, but it is superseded by the token
  chunking baseline below.

Results:

| Chunk Target | Overlap | Avg Score | Passed / 84 | Notes |
| ---: | ---: | ---: | ---: | --- |
| 1,800 chars | 200 chars | 0.6087 | 31 | Best early character-based run; improved known-item and exploratory recall. |
| 2,800 chars | 250 chars | 0.5458 | 24 | Earlier default; weaker than 1,800 on this eval scaffold. |
| 5,000 chars | 350 chars | 0.5401 | 25 | Slightly better exploratory than 2,800, worse overall than 1,800. |
| 10,000 chars | 500 chars | 0.5478 | 22 | Fewer chunks, but lower precision and fewer passes. |

Decision:

- Do not use this character-based chunker as the final baseline.
- The results support a general principle: overly broad chunks hurt precision,
  and moderately precise child chunks work better for this corpus.

## 2026-05-07 - Token-Aware Child Chunk Baseline

Decision:

Use the user-specified token-aware baseline:

```env
CHUNK_SIZE_TOKENS=600
CHUNK_OVERLAP_TOKENS=100
MIN_CHUNK_TOKENS=120
MAX_CHUNK_TOKENS=900
TABLE_CHUNK_TARGET_TOKENS=700
TABLE_CHUNK_MAX_TOKENS=900
TABLE_REPEAT_HEADER=true
CREATE_METADATA_CHUNKS=true
PARENT_SECTION_MAX_TOKENS=1800
ANSWER_CONTEXT_NEIGHBOR_CHUNKS=1
HYBRID_TOP_N_LEXICAL=80
HYBRID_TOP_N_VECTOR=80
HYBRID_TOP_N_METADATA=30
ANSWER_MAX_CHUNKS=12
```

Core architecture:

- Retrieve precise child chunks.
- Include one metadata chunk for every document.
- Use parent/neighbor expansion only for answer context.
- Do not embed full parent sections as the primary retrieval unit initially.

Reasoning:

- 600-token chunks are precise enough for citations and broad enough to preserve
  QMS evidence context.
- 100-token overlap reduces boundary misses without creating too many duplicate
  near-identical chunks.
- Metadata chunks make sparse/empty documents retrievable by filename, document
  code, revision, status, and warnings.
- Neighbor expansion gives synthesis enough context after retrieval without
  sacrificing retrieval precision.

Tests added or planned:

- Metadata chunk is created for every document.
- Metadata-only/sparse docs remain retrievable by filename and document code.
- Prose chunks stay within `MAX_CHUNK_TOKENS`.
- Short sections below `MIN_CHUNK_TOKENS` merge with adjacent content unless
  metadata-important.
- `search_text` includes a compact metadata preamble.
- Raw `text` remains clean for citation display.
- Chunk IDs are stable across repeated indexing.
- Selected answer chunks can expand to neighboring chunks from the same parent
  section.

## 2026-05-07 - Table Chunking

Decision:

- Use `TABLE_CHUNK_TARGET_TOKENS=700`.
- Use `TABLE_CHUNK_MAX_TOKENS=900`.
- Repeat compact table headers in every split table chunk.
- Split tables by row groups only; never split a row across chunks.
- Keep chunk-level citation identity for table chunks so hits from the same table
  do not collapse into one citation.

Worker evidence:

- A table-focused worker inspected table-heavy `VVAM` and `RSK` normalized
  Markdown and wrote scratch results under `/private/tmp/qms-chunk-worker-3/`.
- Simulated table-only chunk counts for extracted table docs:
  - 500-token target: about 13.3k chunks
  - 700-token target: about 9.3k chunks
  - 900-token target: about 7.2k chunks
- 500-token chunks were too fragmented and created high repeated-header
  overhead.
- 900-token chunks reduced chunk count but mixed more unrelated requirements and
  risks, weakening citation precision.
- 700-token target was the best balance of precision, context, and overhead.

Important implementation note:

- Repeat compact semantic headers, not every bulky extracted preamble row.
- For VVAM, prefer a compact `Columns:` line with requirement/proof/pass-fail
  fields.
- For RSK, keep enough group/header context that `P1`, `P2`, `Residual Risk`,
  and `Acceptable?` are interpretable.

Tests to keep:

- Table rows are not split across chunks.
- Headers repeat in split table chunks.
- Header rows do not count as data row ranges.
- Oversize single rows stay intact and are marked/allowed.
- Same-table chunks remain separately citeable by chunk ID or row range.

## 2026-05-07 - Parallel Metadata Coverage Findings

Worker output:

- `/private/tmp/qms-chunk-worker-2/current-code-index/metadata_retrieval_summary.json`
- `/private/tmp/qms-chunk-worker-2/current-code-index/service_query_probe.json`
- `/private/tmp/qms-chunk-worker-2/metadata_impacted_eval_cases.json`

Findings:

- A scratch rebuild before the metadata-only fix represented only `165` ingested
  documents and skipped `24` sparse/empty-body DOCX files before chunking.
- Skipped files included important eval targets such as `IFU-MX1`, `TRA-025`,
  `TRA-026`, `VVPR-P01-179`, `VVPR-P01-214`, and `MEMO-P01-655`.
- `IFU-MX1 - MX1 Instructions for Use_L.docx` was parsed as a long fallback ID
  rather than canonical `IFU-MX1`.
- The previously built SQLite artifact used the older chunk schema and lacked
  `kind`, `search_text`, `parent_section_id`, and `token_count`.

Decision and implemented fix:

- Preserve sparse/empty-body DOCX files as metadata-only documents instead of
  skipping them.
- Add explicit IFU product-style ID parsing for `IFU-MX1`.
- Clear stale normalized Markdown files on ingest so renamed/misparsed documents
  do not survive a rebuild.
- Add SQLite schema-current status with `requires_rebuild` for stale local DBs.

Verification:

- Main rebuild now reports `189` documents, `24` metadata-only documents, and
  `7,778` chunks.
- Targeted tests cover `IFU-MX1` parsing, metadata-only retrieval, and schema
  status.

Eval cases expected to benefit:

- `qms_known_item_retrieval_003`
- `qms_known_item_retrieval_006`
- `qms_known_item_retrieval_010`
- `qms_known_item_retrieval_011`
- `qms_exploratory_search_007`
- `qms_exploratory_search_011`
- `qms_revision_change_tracking_007`
- `qms_enumeration_counting_005`
- `qms_enumeration_counting_006`

## 2026-05-07 - Answer Context Expansion Findings

Worker output:

- `/private/tmp/qms-chunk-worker-4/top_failures.tsv`
- `/private/tmp/qms-chunk-worker-4/best_run_context_signals.tsv`
- `/private/tmp/qms-chunk-worker-4/context_recommendations.md`

Decision:

Use:

```env
ANSWER_CONTEXT_NEIGHBOR_CHUNKS=1
PARENT_SECTION_MAX_TOKENS=1800
```

Reasoning:

- One neighbor chunk catches adjacent headings, short results, and nearby table
  context without frequently dragging in unrelated QMS evidence.
- A parent cap of `1800` tokens covers most useful nearby context while avoiding
  the noisier behavior expected from `2400` by default.
- Many zero-score cases are retrieval/routing failures, so answer-context
  expansion should be treated as a precision aid after a correct source is found,
  not as a replacement for SQL, metadata, or reference-following fixes.

Likely beneficiaries:

- `qms_content_extraction_synthesis_004`
- `qms_known_item_retrieval_002`
- `qms_content_extraction_synthesis_010`
- `qms_content_extraction_synthesis_001`
- `qms_compliance_cross_reference_006`
- `qms_cross_document_analysis_001`

Tests to keep:

- Neighbor expansion stays within the same document and revision.
- Expansion is ordered and deduped.
- Expansion does not cross parent-section boundaries.
- Parent context is included only when under the configured cap.
- Citations remain tied to retrieved chunk/document identity.

## 2026-05-07 - Eval Baseline Status

Current full-suite local eval command:

```bash
uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --hash-embeddings --mode local
```

Most recent pre-token-aware report:

- `docs/eval-runs/2026-05-07-011802.md`
- Average score: `0.5478`
- Passed: `22 / 84`

Interpretation:

- This is a baseline failure report, not a quality claim.
- Known gaps:
  - exact matching for sparse/metadata-only docs
  - SQL-backed enumeration answer formatting
  - multi-hop trace expansion
  - table row citation precision
  - hosted/OpenAI embedding evals versus deterministic hash smoke evals

Next eval requirement:

- Rebuild the index using the token-aware baseline.
- Rerun the full 84-case eval.
- Record a new report and compare against the character-based baseline.

## 2026-05-07 - Token Baseline Rebuild

Command:

```bash
uv run ingest-qms
uv run build-qms-index --hash-embeddings
```

Observed artifact counts after implementing the token-aware chunker:

| Artifact | Value |
| --- | ---: |
| Real DOCX records | 189 |
| Skipped empty records | 0 |
| Metadata-only records | 24 |
| Token-aware chunks | 7,778 |
| Embedding dimensions in manifest | 3,072 |
| FAISS index type in manifest | `IndexFlatIP` |

Decision impact:

- We now retain all real DOCX files, including sparse files, as searchable
  metadata records.
- Metadata-only records are included in SQLite FTS, local vector metadata, and
  reranking candidates.
- The deterministic local review index uses hash embeddings with the same
  configured `3072` dimensions for smoke/eval portability. The production-quality
  build path still calls OpenAI `text-embedding-3-large` with
  `dimensions=3072`.

Tests:

- `uv run pytest evals/test_ingest.py evals/test_chunking.py evals/test_sqlite_store.py evals/test_config.py evals/test_faiss_store.py -q`
- Result: `18 passed`.

Follow-up:

- Run the full 84-case eval against this rebuilt token baseline and capture the
  new report in `docs/eval-runs/`.

## 2026-05-07 - Token Baseline Full Eval

Command:

```bash
uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --hash-embeddings --mode local
```

Report:

- `docs/eval-runs/2026-05-07-014026.md`

Result:

| Metric | Value |
| --- | ---: |
| Passed | 25 / 84 |
| Average score | 0.5210 |
| Known-item retrieval | 8 / 12 passed, 0.7708 avg |
| Content extraction / synthesis | 6 / 12 passed, 0.6833 avg |
| Compliance cross-reference | 0 / 12 passed, 0.1875 avg |

Interpretation:

- The token-aware chunking rebuild improved pass count over the immediate
  pre-token deterministic hash eval (`22 / 84`) but has not matched the best
  early character-chunk smoke eval (`31 / 84`). Because the character eval used
  an older corpus/index shape and lacked metadata-only coverage, it is a useful
  signal but not the architecture winner.
- Known-item retrieval is the current strongest category. The metadata chunks
  are helping exact/filename-style lookups.
- Compliance, cross-document, and revision categories need deterministic
  implementation work more than chunk-size tuning: SQL/list routing,
  reference-following, revision-chain diffing, table row citations, and answer
  formatting.
- The current hash-embedding eval is a smoke/regression proxy. The highest
  quality path still needs the OpenAI 3072-dimensional index and hosted/file
  search comparison before making final retrieval-quality claims.

Planned fix classes from the report:

- multi-hop compliance and reference expansion
- exploratory recall and grouping
- SQL-backed enumeration formatting
- cross-document trace expansion
- revision chain and obsolete/signed handling
- table/section extraction and citation precision
- known-item exact/metadata routing

## 2026-05-07 - Parallel Prose Chunking Matrix

Worker output:

- Scratch report:
  `/private/tmp/qms-chunk-worker-1/matrix-current-eval/matrix_current_eval_summary.json`
- The worker did not modify tracked repo files.

Matrix tested:

| Prose Size | Overlap Values | Passed / 84 | Avg Score |
| ---: | --- | ---: | ---: |
| 450 | 75, 100, 150 | 19 | 0.4710 |
| 600 | 75, 100, 150 | 19 | 0.4710 |
| 750 | 75, 100, 150 | 19 | 0.4710 |
| 900 | 75, 100, 150 | 19 | 0.4710 |

Decision:

- Keep the current prose baseline: `CHUNK_SIZE_TOKENS=600` and
  `CHUNK_OVERLAP_TOKENS=100`.
- The matrix gives no evidence that prose size/overlap tuning is the next useful
  optimization target.

Important caveats:

- The worker's temp matrix used intermediate code and reported `7,730` chunks,
  `7,195` table chunks, `370` prose chunks, and `165` metadata chunks.
- The main branch index was rebuilt after metadata-only ingestion and IFU parsing
  fixes; the current committed artifact has `7,778` chunks and `189` metadata
  chunks.
- Because tables dominate the corpus, changing prose chunk size had little or no
  effect on generated chunks in the matrix.

Interpretation:

- Current retrieval failures are not primarily caused by the 600-token prose
  baseline.
- The higher-leverage work is now deterministic routing and evidence handling:
  SQL enumeration, revision-chain logic, table row citations, multi-hop
  reference following, and OpenAI 3072-dimensional embedding/hosted retrieval
  comparison.

## 2026-05-07 - Bug Decisions Captured

- `search-evals` console script initially failed because the wheel only included
  `src/agent`; fixed by including `evals` in package build config.
- ECR filename parsing initially treated `BOM-055 Rev G` as the ECR document
  revision; fixed by preferring the filename suffix revision (`_A-signed`) while
  preserving `Rev G` in the title.
- FTS queries containing hyphenated document IDs can be parsed as column syntax;
  fixed by quoting tokenized FTS terms and falling back to `LIKE` when needed.

See also:

- `docs/bugs/bugs.md`
- `docs/eval-runs/`
- `DESIGN.md`
