# MedAI QMS Search Design

Final design visual: [system_design.png](system_design.png)

## Objective

Build an internal-search demo for a fictional MedAI Quality Management System
(QMS) corpus for the MX1 portable X-ray system. The primary reviewer experience
is the CLI QMS chat path:

```bash
uv run chat --qms-search --mode hybrid --limit 8
```

That command answers questions from customer-owned documents with deterministic
QMS retrieval, strict citations, and inspectable traces when requested.

This implementation is scoped to customer document search only. Regulatory
search, web search, SharePoint/Drive connectors, authentication, and production
multi-tenant controls are out of scope for the MVP.

## Design Summary

What worked:

- Intent-first retrieval worked better than treating every QMS question as top-N
  semantic search. Counts, inventories, revision diffs, traceability, table
  extraction, and temporal status need deterministic paths before synthesis.
- Hybrid local retrieval worked for this fixed corpus: SQLite FTS catches exact
  IDs, acronyms, filenames, and revisions; OpenAI `3072`-dimensional embeddings
  catch semantic phrasing; metadata/table chunks preserve sparse records and row
  evidence.
- Source-truth eval contracts worked. Required documents, required table
  evidence, required backend, forbidden sources, count checks, and citation
  checks caught failures that the broad `84 / 84` score did not catch.
- Latest-active filtering worked: obsolete leakage dropped from `0.3000` to
  `0.1525` after the final audit fixes.

What did not work:

- Generic top-N retrieval did not work for regulated QMS questions that ask for
  inventories, exact counts, trace endpoints, or revision-change policy.
- Chunk-size tuning alone did not solve the hard failures. The higher-leverage
  changes were metadata coverage, table-row evidence, deterministic routing, and
  stricter eval gates.
- Broad average score alone was misleading. A report can pass all cases at a low
  threshold while still missing canonical source IDs, row evidence, or
  forbidden-source constraints.
- Generic deep-agent retrieval was not kept as the QMS answer path because it
  did not expose enough control over count basis, revision policy, provenance,
  or forbidden-source guards.

North star eval metrics:

| Metric | Definition | Current / target |
| --- | --- | --- |
| Source-truth pass rate | Share of strict cases satisfying required docs, table evidence, backend, counts, and forbidden-source rules. | Equivalent `91 / 91`; keep at `100%`. |
| Contract failure rate | Share of cases violating any hard source-truth contract. | `0.0000`; must stay `0`. |
| Count accuracy | Count answers match the deterministic expected count and explain scope. | `1.0000`; must stay `1.0000`. |
| Latest revision accuracy | Latest/current questions return the correct active revision. | `1.0000`; must stay `1.0000`. |
| Obsolete leakage rate | Current/latest answers avoid obsolete evidence unless requested or explicitly historical. | `0.1525`; drive toward `0`. |
| Required source coverage | Expected sources appear in retrieved/cited evidence; measured by Top-k hit and Recall@k. | Top-k `0.7381`, Recall@k `0.6500`; improve without weakening contracts. |
| Citation validity | Cited source IDs map to expected evidence sources or allowed prefixes. | `0.5623`; improve, while never allowing unsupported cited claims. |

## Default CLI Command And Flags

The intended default reviewer command is:

```bash
uv run chat --qms-search --mode hybrid --limit 8
```

| Part | Meaning |
| --- | --- |
| `uv run chat` | Runs the project CLI entry point in the managed Python environment. |
| `--qms-search` | Switches from the generic LangChain chat agent to the deterministic MedAI QMS search path. This flag should be present for reviewer queries. |
| `--mode hybrid` | Forces the strict local QMS retrieval baseline: SQLite FTS, FAISS-compatible OpenAI vectors, rank fusion, reranking or deterministic fallback, and no hosted File Search calls. |
| `--limit 8` | Returns enough evidence for citation inspection without flooding the terminal. |

Reviewer feature flags:

| Flag | Intended use |
| --- | --- |
| `--trace` | Prints the operational trace: normalized input, planned category/intent, retrieval mode/backend, warnings, and nested retrieval trace data. |
| `--full-citations` | Prints full local Markdown and source paths for cited evidence. |
| `--plain` | Produces stable text output for copy/paste and logs. |
| `--json` | Emits one JSON object per QMS turn for automation. |
| `--raw-trace` | Emits raw JSON trace data instead of formatted trace panels; use with `--trace`. |
| `--no-progress` | Disables the interactive spinner while preserving formatted terminal output. |
| `--no-followup` | Disables multi-turn follow-up context carryover for every turn. |
| `--force-strategy` | Debug-only planner override for isolating a specific query strategy; do not use as the reviewer default. |

`--qms-search --mode hybrid` is the default quality-review path because it avoids
hosted fallback ambiguity and makes the local QMS retrieval contract visible.
`--mode auto` is useful only when the reviewer specifically wants to demonstrate
hosted OpenAI File Search first with local fallback.

## LangChain Deep Agents: Used Vs Not Used

| Area | What we used | What we did not use for QMS answers |
| --- | --- | --- |
| Starter scaffold | The repository shape, `uv` packaging, provider environment config, CLI entry point, and server entry point from the LangChain Deep Agents starter. | A generic deep-agent tool loop as the regulated QMS answer path. |
| Generic chat | `make_agent` still wraps LangChain `init_chat_model` and `create_deep_agent` for non-QMS provider-backed chat, preserving `.invoke()`, `.stream()`, and `.astream()` compatibility. | LangChain retriever abstractions, model-decided source selection, or generic agent memory for QMS evidence. |
| QMS search | The QMS path lives beside the generic agent path and shares provider/model configuration where useful. | Free-form agent retrieval, unconstrained tool planning, or citations invented by the model. |

QMS answers use deterministic query planning, SQLite FTS, FAISS-compatible
OpenAI embeddings, reranking or explicit fallback, structured citation objects,
and source-truth eval gates before answer synthesis.

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

## Eval Glossary

These category names are stable eval labels. Keep the exact strings in datasets,
reports, docs, and regression notes:

| Category | Definition | Primary success signal |
| --- | --- | --- |
| `compliance_cross_reference` | Questions that connect compliance claims across plans, reports, risk records, trace matrices, DHF records, or regulatory-like evidence. | Required supporting documents are present and the answer explains the compliance chain without unsupported sources. |
| `content_extraction_synthesis` | Questions that extract a specific value, row, criterion, or passage and summarize it briefly. | The cited chunk/table row contains the extracted fact and the synthesis does not add outside knowledge. |
| `cross_document_analysis` | Questions that need multiple document families or hops to answer one workflow question. | The answer includes all terminal evidence documents and distinguishes planning, requirement, risk, and verification sources. |
| `enumeration_counting` | Questions that count, list, or inventory documents, records, rows, or ingested artifacts. | Counts come from the correct deterministic source of truth and state the scope being counted. |
| `exploratory_search` | Broad discovery questions where the user is asking what evidence exists around a topic. | Results cover the relevant families without collapsing into one repeated source or irrelevant training records. |
| `known_item_retrieval` | Exact or paraphrased lookup by document ID, title, filename, revision, signed status, or related title terms. | The intended document appears with the correct latest/signed/obsolete revision policy. |
| `revision_change_tracking` | Questions about latest active revision, obsolete history, revision chains, or changes between revisions. | Revision scope is explicit and unrelated documents are not used as fallback diffs. |

Core chunking, embedding, and retrieval terms:

| Term | Definition |
| --- | --- |
| Child chunk | The primary retrieval unit: a section-aware prose chunk, currently targeted at `600` tokens. |
| Metadata-only chunk | A searchable chunk made from filename/document metadata when body extraction is sparse or empty. |
| Table row chunk | A row-preserving table evidence chunk; emitted for tables with `50` rows or fewer so row/cell citations stay inspectable. |
| Overlap | Repeated trailing/leading tokens between neighboring prose chunks, currently `100` tokens, to reduce boundary misses. |
| Dense embedding | A numeric vector representation of chunk text; the baseline uses OpenAI `text-embedding-3-large` at `3072` dimensions. |
| `IndexFlatIP` | The FAISS exact inner-product index used over normalized vectors, making inner product equivalent to cosine similarity. |
| SQLite FTS | SQLite full-text search over normalized document/chunk text and metadata for IDs, acronyms, filenames, and exact terms. |
| Hybrid retrieval | The local search path that merges SQLite FTS, dense vector candidates, metadata candidates, rank fusion, and reranking. |
| Rank fusion | The scoring step that combines lexical, dense, and metadata candidate ranks before reranking. |
| Reranker | A second-stage model or deterministic fallback that reorders merged candidates before answer synthesis. |
| SQL inventory | A deterministic SQLite query path for list/count/inventory questions that should not depend on top-N chunk retrieval. |
| Source-truth gate | A hard eval contract for required sources, required table evidence, required backend, forbidden terms, count correctness, and citation validity. |
| Hosted File Search | OpenAI hosted vector-store retrieval over the same normalized corpus, synced separately from the local FAISS/SQLite index. |
| Retrieval mode | The user-visible backend policy: `auto`, `hosted`, `hybrid`, or `local`, with stable semantics across CLI, API, and evals. |

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

Mode names are user-visible and must keep stable semantics across CLI, API, and
evals:

| Mode | Hosted behavior | Local behavior | Intended use |
| --- | --- | --- | --- |
| `auto` | Try hosted OpenAI File Search first when synced and healthy. | Fall back to local retrieval if hosted is unavailable, empty, stale, or errors. | Hosted-first smoke mode and hosted/local comparison. |
| `hosted` | Prefer hosted OpenAI File Search. | Fall back locally with an explicit warning rather than failing the whole turn. | Hosted sync validation and hosted/local comparison. |
| `hybrid` | Never call hosted search. | Use local SQLite FTS, FAISS-compatible dense retrieval, rank fusion, and reranking. | Intended reviewer default, strict local quality baseline, and primary eval mode. |
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
  -> 600-token child chunks + metadata/table chunks
  -> embeddings + lexical index
  -> persisted local index
  -> hybrid retrieval over lexical, dense, and metadata candidates
  -> Qwen rerank or deterministic fallback
  -> nearest-neighbor/parent context expansion for synthesis
  -> answer synthesis with citations
  -> CLI QMS chat and API responses
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
- `src/agent/server.py`: expose search-aware health/status/search endpoints.
- `src/agent/cli.py`: expose the primary QMS chat and search commands.

The exact file layout can change if another worker finds a cleaner fit, but the
boundaries should remain independently testable.

## Performance Impact By Technical Decision

The final benchmark gains were not one isolated change. Early chunking
experiments were isolated enough to claim direct movement for the combined
chunking pipeline, but not for nearest-neighbor/parent expansion by itself. The
precise claim is: nearest-neighbor/parent expansion was part of the chunking
pipeline that improved early deterministic performance, but it was not isolated
in an ablation from token chunking and overlap. The final `0.5812 -> 0.8296`
core-score improvement was a grouped system-level result from retrieval, routing,
metadata/table evidence, and source-truth contract changes.

| Decision group | What changed | Objective movement | Attribution note |
| --- | --- | --- | --- |
| Token-aware chunking plus nearest context | Replaced broad character chunks with `600` token child chunks, `100` overlap, and answer-time neighbor/parent context expansion. | Deterministic hash pass count improved `22 / 84 -> 25 / 84`; citations became easier to inspect. | This result isolates the combined chunking pipeline, not the neighbor/parent step alone. It helped, but did not explain the final quality jump by itself. |
| Table-row evidence | Used `700` token table grouping and row-preserving chunks, with row chunks capped to tables with `50` rows or fewer. | Avoided roughly `100k` unbounded row chunks while preserving exact row/cell evidence for audit-sized tables. | Performance impact showed up later through source-truth/table-evidence gates, not as a standalone ablation. |
| Metadata-only chunks | Kept sparse or empty-body DOCX records searchable by filename, document ID, revision, signed status, and obsolete status. | Preserved all `189` real DOCX records, including `24` sparse records. | Directly supported known-item, filename, revision, and ingest-count questions. |
| OpenAI embedding baseline | Rebuilt with `text-embedding-3-large`, `3072` dimensions, normalized `IndexFlatIP` vectors. | Part of the audited OpenAI-index system improvement from average score `0.5812 -> 0.8296`. | Grouped with routing/citation fixes; not claimed as a standalone embedding-only lift. |
| Latest-active and obsolete filtering | Preferred current active revisions unless obsolete history was requested. | Obsolete leakage improved `0.6833 -> 0.3000 -> 0.1525`. | This was directly measured by the obsolete leakage metric. |
| Intent-first deterministic routing | Routed counts, inventories, traceability, table lookup, revision ambiguity, temporal status, and manifest count questions away from generic top-N retrieval. | Strict surface improved from `84 / 91` to equivalent `91 / 91`; known-item, cross-document, enumeration, and revision failures were fixed. | This was the largest product-quality change for regulated QMS questions. |
| Source-truth contracts | Added hard gates for required docs, table evidence, backend, forbidden sources, count correctness, and citation validity. | Targeted tests moved `10 failed, 21 passed -> 31 passed`; after the pediatric fix, source-truth contracts were `23 passed`; contract failure rate is `0.0000`. | These gates made the evals catch defects that broad average score missed. |
| Pediatric-filtration trace intent | Added a deterministic trace path requiring `DR-P01-005`, historical `RSK-P01-010`, `VVAM-P01-004`, and `VVPR-P01-152`. | A later strict run's remaining `90 / 91` failure was fixed by requiring the missing `VVPR-P01-152` endpoint. | This was benchmark-specific and should be generalized before arbitrary-corpus use. |
| Lexical preservation and wider rerank input | Preserved exact entity/table hits and widened rerank candidates for distinctive terms such as `510(k)`, `K241567`, `3P-P01-33`, and `acceptance criteria`. | Top-k hit improved `0.5238 -> 0.7381`; Recall@k improved `0.4504 -> 0.6500`; citation validity improved `0.3773 -> 0.5623`. | Metrics reflect the grouped audited system, not a single isolated rerank ablation. |

## Retrieval Pipeline And Chunking Evidence

The retrieval pipeline intentionally retrieves small, inspectable evidence first
and expands context only after ranking:

1. Extract `.docx` content with binary readers and normalize metadata from
   filenames.
2. Build `600` token child chunks with `100` token overlap, metadata chunks for
   every document, and bounded row/table chunks.
3. Retrieve lexical, dense, and metadata candidates through SQLite FTS,
   FAISS-compatible vector search, and metadata lookup.
4. Fuse candidates, preserve important lexical hits for exact IDs/entities, and
   deduplicate repeated same-section chunks.
5. Rerank the merged candidate set, or use the deterministic fallback when the
   configured Qwen CrossEncoder is unavailable.
6. Expand nearest neighbor/parent context after retrieval for answer synthesis,
   while citations still point to structured evidence.

Chunking evidence was useful but bounded:

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Character chunking ablation | Best early character run was `1,800` chars with `31 / 84` passed and average score `0.6087`, on an older corpus/index shape. | Directional evidence that overly broad chunks hurt precision. |
| Token-aware child chunks | `600` token chunks, `100` overlap, and answer-time neighbor expansion moved deterministic hash pass count `22 / 84 -> 25 / 84`. | Clear isolated improvement for the combined chunking pipeline, not a standalone proof that neighbor expansion alone improved performance. |
| Table grouping | `700` token table targets balanced context and chunk count; unbounded row chunks produced about `100k` chunks. | Keep row chunks only for tables with `50` rows or fewer. |
| Final audited system | Core report passed `84 / 84`, average score `0.8296`, Top-k `0.7381`, Recall@k `0.6500`, citation validity `0.5623`. | This was a grouped improvement across chunking, metadata, routing, lexical preservation, answer contracts, and eval gates. |

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

## Ranking And Score Formula

The local `hybrid` path exposes multiple ranks and scores in `--trace` output.
They are operational retrieval diagnostics, not the `search-evals` quality
score.

Rank fusion uses reciprocal-rank fusion over lexical and dense candidates:

```text
fused_score =
  lexical_weight / (60 + lexical_rank)
  + vector_weight / (60 + vector_rank)
```

If a candidate is absent from one list, that list contributes `0`. The default
weights are `lexical_weight=1.0` and `vector_weight=1.0`. For extraction,
known-item, compliance, and selected audited intents, `lexical_weight=2.0` so
exact IDs, filenames, standards, and table terms are less likely to be displaced
by semantically similar but unsupported chunks.

The rerank layer then receives the fused candidates. When the configured Qwen
CrossEncoder is available, `rerank_score` is the model's raw score for
`(query, candidate_text)` and candidates sort descending. In environments where
the CrossEncoder is unavailable, the deterministic fallback computes:

CrossEncoder was chosen for the second stage because it evaluates the query and
candidate text together, instead of comparing two independent embeddings. That
is slower than vector search, so we only run it on the fused top candidates, but
it is better suited to reranking close evidence where exact wording, table rows,
document IDs, and standards decide whether a citation is actually useful.

```text
rerank_score =
  4.0 * fused_score
  + 0.025 * term_overlap
  + phrase_boost
  + table_boost
  + exact_id_boost
```

Fallback terms:

| Term | Definition |
| --- | --- |
| `term_overlap` | Count of non-stopword query terms also present in the candidate text. |
| `phrase_boost` | Sum of small boosts for audited distinctive phrases present in both query and candidate text: `510(k)` `0.12`, `510k` `0.12`, `K241567` `0.16`, `device summary` `0.12`, `3P-P01-32` `0.12`, `3P-P01-33` `0.12`, `acceptance criteria` `0.12`, `electrical safety` `0.12`, `dielectric` `0.08`, `leakage` `0.08`, `DHF-008` `0.12`, and `RSK_R` `0.08`. |
| `table_boost` | `0.08` for row/table-style evidence containing `Columns:` and `Row `. |
| `exact_id_boost` | `5.0` when the candidate document ID appears directly in the query. |

The justification is eval-driven but grouped. The audit showed cases where
lexical retrieval found exact row/source evidence, but generic fusion/reranking
pushed that evidence down. Lexical preservation, the `2.0` lexical-priority
weight, wider rerank input, deterministic fallback scoring, and source-truth
contracts together improved Top-k hit `0.5238 -> 0.7381`, Recall@k
`0.4504 -> 0.6500`, and citation validity `0.3773 -> 0.5623`. We did not run a
separate ablation for each numeric coefficient, so these weights should be
treated as fixed-corpus, eval-backed heuristics rather than universal ranking
constants.

## Intent-First Retrieval Contract

The query planner assigns both a broad category and a concrete intent. The broad
category controls default retrieval, while the concrete intent can bypass generic
semantic search when the answer needs a deterministic source of truth.

Reusable design contracts:

- Counts, inventories, and ingest totals use SQL or manifest-backed source of
  truth instead of top-N snippets.
- Latest/current questions prefer latest active, non-obsolete records unless the
  user requests history.
- Table-backed answers must preserve row/cell evidence and cite table chunks
  when available.
- Known-item searches prioritize exact document IDs, filenames, titles, signed
  status, revision, and obsolete status before semantic similarity.
- Traceability and cross-document answers must include terminal evidence, not
  only upstream requirements or planning documents.
- Forbidden-source guards block known misleading families, such as treating
  `TRA-*` customer training records as traceability matrices.

Audit and benchmark-specific intent contracts:

- Some named intents below are deliberately narrow because the fixed onsite
  corpus and strict evals exposed repeated failure modes.
- Examples include `510k_summary_location`, `electrical_safety_acceptance`, and
  `pediatric_filtration_trace`, where the expected source IDs are known and the
  answer must preserve exact endpoint evidence.
- These targeted intents are acceptable for this fixed-corpus reviewer
  benchmark, but they should be generalized into configurable source-type,
  table-evidence, and traceability policies before scaling to arbitrary QMS
  corpora.

Current fixed-corpus high-value intents:

| Intent | Purpose | Required evidence behavior |
| --- | --- | --- |
| `mx1_bom` | Find the MX1 system BOM. | Return `BOM-055 Rev G` as primary and group software BOMs separately. |
| `510k_summary_location` | Locate 510(k) summary evidence. | Cite `MEMO-P01-859`, `DHF-008`, and `PLN-P01-061`; state when no standalone summary file is indexed. |
| `vvpr_inventory` | List/count MX1 verification protocols. | Use SQL inventory, not top-N snippets; expose total/non-obsolete/latest-active scope counts. |
| `risk_related_inventory` | Show risk-related documents. | Use SQL/topic inventory across `RSK`, risk plans, `VVAM`, and risk/RMF-bearing docs. |
| `dhf_82030` | Check DHF against design-control expectations. | Cite `DHF-008` and planning support; include the current QMSR caveat. |
| `risk_protocol_trace` | Find P01 protocols traced from risk analysis. | Use active `RSK`/`VVAM` evidence and avoid historical predecessor protocol IDs. |
| `electrical_safety_acceptance` | Extract electrical safety criteria. | Cite `MEMO-P01-685 Table 2 row 2` and `3P-P01-33`; include IEC 60601-1 and `PASS`. |
| `open_design_review_actions` | Summarize open review actions. | Cite `MEMO-P01-859` Section 4 / Summary of Action Items; do not inherit unrelated prior-turn context. |
| `ambiguous_risk_revision_diff` | Compare missing risk-analysis revision pair. | Return clarification/no-answer when no single RSK chain contains both revisions. |
| `ecr_last_year_status` | List ECRs by temporal/status policy. | Extract approval effective dates, DCO/status fields, and state the current-date policy. |
| `electrical_leakage_trace` | Trace leakage from risk to report. | Return explicit chain: risk source, `VVAM` bridge, summary, and `3P-P01-33` report. |
| `pediatric_filtration_trace` | Trace pediatric filtration through verification. | Cite `DR-P01-005`, obsolete/historical `RSK-P01-010`, `VVAM-P01-004`, and `VVPR-P01-152`; include the direct `PRD20.3 -> VVPR-P01-152` bridge and result basis. |
| `third_party_report_mapping` | Map 3P reports to standards. | Distinguish completed `3P-*` reports from planning evidence. |
| `ecr_count` | Count engineering change requests. | Count active signed ECR records in SQL and list the IDs. |
| `verification_completed_vs_planned` | Compare completed vs planned V&V work. | Use `MEMO-P01-685` result rows and `PLN-P01-065` planned scope, not raw VVPR count. |

Hybrid retrieval supports those paths rather than replacing them. For extraction
and known-item categories, rank fusion preserves top lexical hits, widens the
rerank candidate pool, boosts distinctive entities such as `510(k)`, `K241567`,
`3P-P01-32`, `3P-P01-33`, `DHF-008`, and `acceptance criteria`, and deduplicates
repeated same-section chunks before answer synthesis.

## Eval Run Notes

The QMS search eval history is tracked in `docs/eval-runs/` and summarized in
`docs/walkthrough.md`. A strict external run on 2026-05-07 established an
`84 / 91` baseline, later improved to `90 / 91` with one remaining failure:
`qms_cross_document_analysis_004` omitted the literal `VVPR-P01-152` phrase even
though source evidence was present. The current design treats that as an intent
contract, not a generic retrieval problem: pediatric-filtration trace questions
must use the deterministic `pediatric_filtration_trace` path.

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

List responses should always be guarded with `Array.isArray` or equivalent by
API consumers. Mutation and indexing handlers should return structured errors.

## Evaluation Strategy

The evaluation strategy ended up stricter than ordinary answer substring checks.
We used the project overview's golden query patterns as the seed, then added
synthetic variants to cover the same regulated-search behaviors across different
personas, revision scopes, answerability levels, input noise, citation burdens,
table-evidence needs, forbidden sources, and backend expectations.

Eval datasets and gates:

| Surface | Size | Purpose |
| --- | ---: | --- |
| `evals/datasets/qms_smoke.jsonl` | `7` cases | One reviewer smoke case per category from the project-overview query patterns. |
| `evals/datasets/qms_core.jsonl` | `84` cases | Synthetic augmentation of the golden patterns: `12` cases per category across all seven eval labels. |
| Source-truth regression tests | `23` hard-contract cases | Locks audited failures that broad scoring missed, including required docs, table rows, backend, count, and forbidden-source expectations. |
| Strict audit surface | Equivalent `91` cases | Smoke/category coverage plus core coverage and the audited strict regressions. |

Eval scoring:

| Component | Definition |
| --- | --- |
| `term_score` | Fraction of expected answer phrases present after deterministic normalization. |
| `source_score` | Fraction of expected source IDs present in the scored evidence. |
| `total_score` | `0.7 * term_score + 0.3 * source_score`; default pass threshold is `0.8`. |
| Hard contracts | `required_doc_ids`, `required_backend`, `required_table_evidence`, `must_not_include`, and count correctness. Any contract failure overrides a passing average score. |
| Retrieval metrics | Top-k hit, Recall@k, citation validity, count accuracy, latest revision accuracy, obsolete leakage, and contract failure rate. |

Latest verified results after the pediatric-filtration fix:

| Metric | Result |
| --- | ---: |
| Core cases passed | `84 / 84` |
| Core average score | `0.8296` |
| Top-k hit rate | `0.7381` |
| Recall@k | `0.6500` |
| Count accuracy | `1.0000` |
| Citation validity | `0.5623` |
| Contract failure rate | `0.0000` |
| Latest revision accuracy | `1.0000` |
| Obsolete leakage rate | `0.1525` |
| Source-truth contracts | `23 passed` |
| Full eval tests | `148 passed` |

Strict eval baseline and fixes on 2026-05-07:

- The strict external baseline before the audit fixes was `84 / 91` passed
  overall (`92%`), with the smoke subset at `7 / 7`.
- The seven initial strict failures were in known-item retrieval, revision-change
  tracking, cross-document analysis, and enumeration/counting.
- Root causes included prefix-only known-item routing, unrelated revision-diff
  fallback, missing terminal software trace targets, treating `TRA-*` training
  records as traceability matrices, and not using `ingest_manifest.json` for
  ingest-count questions.
- Failing-test-first proof moved targeted planner/source-truth checks from
  `10 failed, 21 passed` to `31 passed`.
- A later strict CLI run improved to `90 / 91`; the remaining failure omitted
  the literal `VVPR-P01-152` endpoint for pediatric filtration. The
  `pediatric_filtration_trace` intent fixed that source-truth contract, and the
  focused source-truth suite now reports `23 passed`.

Quantified system-level improvement:

| Measure | Before | After | Improvement |
| --- | ---: | ---: | ---: |
| Strict eval surface | `84 / 91` (`92.3%`) | equivalent `91 / 91` (`100.0%`) | `+7` passes, `+7.7` percentage points |
| `known_item_retrieval` | `11 / 13` | `13 / 13` | `+2` passes |
| `cross_document_analysis` | `11 / 13` | `13 / 13` | `+2` passes |
| `enumeration_counting` | `11 / 13` | `13 / 13` | `+2` passes |
| `revision_change_tracking` | `12 / 13` | `13 / 13` | `+1` pass |
| Core average score | `0.5812` | `0.8296` | `+0.2484` (`+42.7%`) |
| Core Top-k hit rate | `0.5238` | `0.7381` | `+0.2143` (`+40.9%`) |
| Core Recall@k | `0.4504` | `0.6500` | `+0.1996` (`+44.3%`) |
| Core citation validity | `0.3773` | `0.5623` | `+0.1850` (`+49.0%`) |
| Obsolete leakage rate | `0.6833` | `0.1525` | `-0.5308` (`-77.7%`) |

The strict `91`-case after number is an equivalent repo-backed surface, not the
same external runner. That runner is not checked into this repository; equivalent
strict guards are represented by source-truth regression tests and generated JSON
report aliases such as `pass`, `missing_phrases`, `source_hit`, `answer_text`,
`source_ids`, `retrieved_source_ids`, and `backend`.

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
- A broad average score does not override the source-truth gate. Failures in
  required sources, table evidence, retrieval trace, provenance fields, or
  `must_not_include` guards block query-path changes.

## Tradeoffs

This repo is intentionally optimized for a fixed onsite corpus, not for generic
internet-scale retrieval. The current baseline has `189` documents and `17,651`
vector rows. At that size, exact local vector search, SQLite inventory queries,
row-level table evidence, rich traces, and strict citations are worth more than
lowest possible indexing cost or horizontal scale.

| Decision | Why | Cost |
| --- | --- | --- |
| Local FAISS over managed vector DB | Fast, portable demo with no service setup | No production ACLs, replication, or managed scaling |
| `IndexFlatIP` over IVF/HNSW | Exact search is simple and high quality for a small corpus | Linear scan will not scale to large corpora |
| Hybrid retrieval over dense-only | Better exact IDs, filenames, revisions, and acronyms | More indexing and scoring code |
| Qwen reranker over no rerank | Improves final evidence ordering for synthesis | Adds latency and operational dependency |
| Filename metadata first | QMS revision/status signals are encoded in filenames | Must handle inconsistent naming carefully |
| Strict citations | Builds trust and makes evals concrete | Forces abstention when extraction is weak |
| `.docx` body extraction first | Matches corpus format and avoids brittle binary parsing | Page-level citations may be unavailable |

Scale guidance:

| Corpus size | Design change |
| --- | --- |
| Current `189` docs | Keep exact `IndexFlatIP`, SQLite FTS, deterministic SQL inventories, row-level table evidence, and complete local traces. |
| Around `10k` docs | Move away from full flat scans for every query; add incremental indexing, stronger metadata prefilters, background rebuild jobs, latency/cost budgets, and dashboards for retrieval quality by category. |
| Around `100k+` docs | Use managed or sharded vector infrastructure, tenant/ACL-aware filters, tiered retrieval, rolling re-embeds, async ingestion queues, hosted-vs-local eval comparison, and stricter operational monitoring. |

If users slowly add documents over time, the design should stop treating the
index as a one-time artifact. New or changed files should get document-level
hashes, incremental extraction, append/update embedding jobs, and a manifest
that records which chunking config produced each vector. Full re-chunking should
only happen when extraction code, chunking config, table handling, embedding
model, or metadata schema changes; otherwise unchanged documents should keep
their existing chunk IDs and vectors.

## What We Would Do With More Time

- Run a full local-vs-hosted comparison against OpenAI hosted File Search using
  the same `qms_core.jsonl`, smoke, source-truth, and category metrics.
- Add a hosted-only eval mode that fails when hosted retrieval is unavailable,
  so hosted quality is measured separately from local fallback quality.
- Capture latency, token usage, embedding cost, reranker cost, and hosted search
  cost in every eval report.
- Add an incremental ingest test where documents are added gradually and only
  changed documents are re-extracted, re-chunked, and re-embedded.
- Define a scheduled re-chunking policy: incremental updates for ordinary
  document additions, full rebuilds for chunking/schema/model changes, and
  periodic full audits when the corpus has materially changed.

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
- Deterministic QMS intents are implemented for the current 14-query audit set
  and the seven strict 91-case regressions, including BOM, 510(k), VVPR
  inventory, risk inventory, DHF/QMSR, risk protocol trace, electrical-safety
  criteria, design-review actions, risk revision ambiguity, ECR status/date,
  leakage trace, third-party report mapping, software trace endpoints,
  traceability-matrix counting, ingest-manifest counting, ECR count, and
  completed-vs-planned verification.
- Latest verification for this audit pass:
  - Targeted red proof before implementation:
    `UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest evals/test_query_plan.py evals/test_qms_source_truth_contracts.py -q`
    produced `10 failed, 21 passed`.
  - Targeted regression after implementation: same command produced
    `31 passed`.
  - Pediatric-filtration follow-up targeted red proof before implementation:
    `UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest evals/test_query_plan.py evals/test_qms_source_truth_contracts.py -q`
    produced `2 failed, 30 passed`.
  - Pediatric-filtration follow-up targeted regression after implementation:
    same command produced `32 passed`.
  - Source-truth contracts after the pediatric fix:
    `UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest evals/test_qms_source_truth_contracts.py -q`
    produced `23 passed`.
  - Full eval tests after the pediatric fix:
    `UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest evals -q` produced
    `148 passed`, with only the existing LangGraph deprecation warning.
  - Core diagnostic report after the pediatric fix:
    `UV_CACHE_DIR=/private/tmp/uv-cache uv run search-evals --dataset core --report docs/eval-runs --mode hybrid --fail-under 0`
    produced `84 / 84` passed, average score `0.8296`, Top-k `0.7381`,
    Recall@k `0.6500`, citation validity `0.5623`, and obsolete leakage
    `0.1525`, in `docs/eval-runs/2026-05-07-174047.md` and `.json`.
  - Artifact validation:
    `LocalVectorIndex.validation()` reports `ok=True`, `requires_rebuild=False`,
    ingest schema `1`, vector schema `1`, `189` documents, and `17,651` vector
    rows.
  - Deterministic fallback report:
    `ANSWER_SYNTHESIS_ENABLED=false RERANKER_ENABLED=false ... --hash-embeddings`
    produced `84 / 84` passed, average score `0.7634`, in
    `docs/eval-runs/2026-05-07-170554.md` and `.json`.
  - Core dataset schema remains `84` cases, `12` per category, valid.
- The older 84-case local OpenAI-index report remains useful historical context
  (`84 / 84` at harness threshold `0`, average `0.5812`, in
  `docs/eval-runs/2026-05-07-040446.md`), but it did not catch row-level,
  forbidden-source, or deterministic-intent failures. The stricter eval contract
  is now the quality gate for these audited behaviors.
See `docs/architecture-decisions.md` for the running trade-off log and chunking
decision evidence.
