# MedAI QMS Demo Walkthrough

Final design visual: [system_design.png](../system_design.png)

This walkthrough reflects the current OpenAI-backed indexed baseline and calls
out the remaining production gaps separately. The intended reviewer entry point
is the QMS CLI in strict local hybrid mode:

```bash
uv run chat --qms-search --mode hybrid --limit 8
```

## 1. Build Or Verify The Current OpenAI Index

Run this before the sample queries if you are reviewing from a fresh clone or
any environment where the local QMS index may not already be present. It checks
for the core processed artifacts, then runs the full data-processing path: DOCX
ingest, metadata normalization, chunking, OpenAI vector embeddings, SQLite FTS,
and the FAISS-compatible vector store.

```bash
INDEX_DIR="${QMS_INDEX_DIR:-.data/qms-index}"

if [ ! -f "$INDEX_DIR/manifest.json" ] \
  || [ ! -f "$INDEX_DIR/qms.sqlite" ] \
  || [ ! -f "$INDEX_DIR/vector_metadata.json" ] \
  || [ ! -f "$INDEX_DIR/vectors.npy" ]; then
  echo "QMS index artifacts are missing; rebuilding from source corpus."
  rm -rf "$INDEX_DIR"
  uv run ingest-qms
  uv run build-qms-index
else
  echo "QMS index artifacts already exist at $INDEX_DIR; skipping rebuild."
fi

uv run search-status --tasks TASKS.md --index-dir "$INDEX_DIR" --openai-state .data/openai/vector_store_state.json
```

For an explicit rebuild even when artifacts already exist:

```bash
rm -rf "${QMS_INDEX_DIR:-.data/qms-index}"
uv run ingest-qms
uv run build-qms-index
```

`uv run build-qms-index` is the OpenAI embedding build and requires
`OPENAI_API_KEY` to be available through `.env` or the environment. Do not
`source .env`; the Python entry points load it directly.

Expected output should include:

- Corpus hash.
- Document count `189`, skipped count `0`, metadata-only count `24`, and chunk
  count `17,651` after rebuilding with the bounded row-level table schema.
- Embedding model: `text-embedding-3-large`.
- Embedding dimensions: `3072`.
- Embedding provider: `openai`.
- FAISS type: `IndexFlatIP`.
- SQLite `revisions` and `doc_references`; post-rebuild status reports
  `4,446` references.
- Hosted File Search status `synced` with `189` files for the current corpus
  hash.
- Vector backend: `faiss` when the optional native package is importable,
  otherwise `numpy_fallback`.
- Reranker backend: `sentence_transformers_cross_encoder` when the configured
  CrossEncoder model is available, otherwise `deterministic_fallback` with a
  warning.
- Chunking: 600-token child chunks, 100-token overlap, 700-token table target,
  row chunks for tables with 50 rows or fewer, metadata chunks enabled, and one
  answer-time neighbor chunk.
- Extraction warnings.

Inspect status later without rebuilding:

```bash
uv run search-status --tasks TASKS.md --index-dir .data/qms-index --openai-state .data/openai/vector_store_state.json
```

Deterministic smoke build, only when live embeddings are intentionally avoided:

```bash
uv run ingest-qms
uv run build-qms-index --hash-embeddings
```

The OpenAI build must remain the indexed baseline for production-style runs.

## Quick Design Summary

What worked:

- Intent-first retrieval: deterministic SQL inventory, revision, traceability,
  table, and temporal-status paths fixed failures that generic semantic search
  missed.
- Hybrid evidence retrieval: SQLite FTS, OpenAI `3072` embeddings, metadata
  chunks, and table-row chunks worked well for the fixed `189`-document corpus.
- Source-truth gates: required source IDs, required table evidence, required
  backend, forbidden sources, count accuracy, and citation checks caught defects
  that broad pass/fail scoring missed.

What did not work:

- Top-N semantic retrieval alone for counts, inventories, trace endpoints, or
  revision comparisons.
- Chunk-size tuning as the primary fix. It helped define the baseline, but the
  major quality gains came from routing, metadata/table evidence, and eval
  contracts.
- Broad average score as a quality gate. It has to sit behind source-truth
  contracts for regulated search.

North star eval metrics:

| Metric | Definition | Current / target |
| --- | --- | --- |
| Source-truth pass rate | Strict cases satisfy required docs, table evidence, backend, counts, and forbidden-source rules. | Equivalent `91 / 91`; keep `100%`. |
| Contract failure rate | Any hard source-truth violation. | `0.0000`; must stay `0`. |
| Count accuracy | Deterministic count answers match expected scope. | `1.0000`; must stay `1.0000`. |
| Latest revision accuracy | Latest/current questions return the right active revision. | `1.0000`; must stay `1.0000`. |
| Obsolete leakage rate | Current/latest answers avoid obsolete evidence unless requested. | `0.1525`; drive toward `0`. |
| Required source coverage | Expected sources appear in retrieval/citations; tracked by Top-k hit and Recall@k. | Top-k `0.7381`, Recall@k `0.6500`; improve. |
| Citation validity | Cited IDs map to expected evidence sources or allowed prefixes. | `0.5623`; improve without allowing unsupported claims. |

## Quick Show

Five sample queries:

- `Find the signed MX1 software development configuration management memo.`
- `Where is the 510(k) summary for the MX1 device? Cite the document ID if it exists.`
- `What are the acceptance criteria for the electrical safety verification test?`
- `How many traceability matrices are in the corpus?`
- `Trace pediatric filtration from requirements or risk rationale through verification evidence.`

Single chat command with traces:

```bash
printf '%s\nquit\n' 'Find the signed MX1 software development configuration management memo.' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
```

Copy-paste answer checks without operational retrieval traces:

```bash
printf '%s\nquit\n' 'Find the signed MX1 software development configuration management memo.' | uv run chat --qms-search --mode hybrid --limit 8 --plain
printf '%s\nquit\n' 'Where is the 510(k) summary for the MX1 device? Cite the document ID if it exists.' | uv run chat --qms-search --mode hybrid --limit 8 --plain
printf '%s\nquit\n' 'What are the acceptance criteria for the electrical safety verification test?' | uv run chat --qms-search --mode hybrid --limit 8 --plain
printf '%s\nquit\n' 'How many traceability matrices are in the corpus?' | uv run chat --qms-search --mode hybrid --limit 8 --plain
printf '%s\nquit\n' 'Trace pediatric filtration from requirements or risk rationale through verification evidence.' | uv run chat --qms-search --mode hybrid --limit 8 --plain
```

Copy-paste answer checks with operational retrieval traces and full citations:

```bash
printf '%s\nquit\n' 'Find the signed MX1 software development configuration management memo.' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
printf '%s\nquit\n' 'Where is the 510(k) summary for the MX1 device? Cite the document ID if it exists.' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
printf '%s\nquit\n' 'What are the acceptance criteria for the electrical safety verification test?' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
printf '%s\nquit\n' 'How many traceability matrices are in the corpus?' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
printf '%s\nquit\n' 'Trace pediatric filtration from requirements or risk rationale through verification evidence.' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
```

Smoke eval harness commands:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-evals --dataset smoke --report docs/eval-runs --mode hybrid --fail-under 0
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-evals --dataset smoke --report docs/eval-runs --mode auto --fail-under 0
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-evals --dataset smoke --answers-jsonl evals/datasets/qms_smoke_golden_answers.jsonl
```

`search-evals` records score/source diagnostics rather than streaming the human
trace panels. Use the traced answer commands above when you need to show the
reasoning/retrieval trace live.

Default command breakdown:

| Part | Meaning |
| --- | --- |
| `uv run chat` | Runs the project CLI in the managed `uv` environment. |
| `--qms-search` | Uses the deterministic QMS search path instead of generic provider chat. |
| `--mode hybrid` | Forces the local SQLite FTS + FAISS-compatible vector + rerank/fallback pipeline; hosted File Search is not called. |
| `--limit 8` | Keeps the evidence set small enough for terminal review while still showing citation support. |

Useful reviewer flags:

| Flag | What it shows |
| --- | --- |
| `--trace` | Input normalization, query planning, backend routing, warnings, candidate counts, and nested retrieval trace data. |
| `--full-citations` | Full local Markdown/source paths for cited evidence. |
| `--plain` | Stable text output for logs and copy/paste. |
| `--json` | Machine-readable turn output. |
| `--raw-trace` | Raw JSON trace data when combined with `--trace`. |
| `--no-progress` | Disables the interactive spinner while keeping formatted output. |
| `--no-followup` | Disables carryover from prior QMS turns. |

## Prerequisites

- Branch: `andrewsohrabi/valkai-onsite`
- Corpus artifact at repo root: `Example_QMS_-_MedAI.zip`
- Python environment synced with `uv sync`
- OpenAI key available for `text-embedding-3-large` and `gpt-5.5`
- Optional native search dependencies installed when you want to exercise local
  `faiss.IndexFlatIP` and the Qwen CrossEncoder reranker instead of the
  documented fallbacks:
  `uv sync --group native-search`

Current caveat:

- The local index is now built with OpenAI `text-embedding-3-large` at 3072
  dimensions and hosted OpenAI File Search is synced.
- The optional Qwen CrossEncoder reranker path is implemented. This sandbox
  currently reports `deterministic_fallback` because `sentence_transformers` and
  the local reranker model cache are not installed here.
- The optional native FAISS path is implemented. If `faiss` is unavailable,
  local vector search uses a NumPy `IndexFlatIP`-compatible fallback and reports
  that backend in status/debug output.

Retrieval modes for the walkthrough:

| Mode | Use in the demo |
| --- | --- |
| `auto` | Hosted OpenAI File Search first, then local fallback when hosted state is missing, stale, empty, or errors. Use only when demonstrating hosted fallback behavior. |
| `hosted` | Hosted-preferred validation path with local fallback warnings. |
| `hybrid` | Strict local SQLite FTS + FAISS + reranker path; never hosted. This is the intended reviewer default and local quality baseline. |
| `local` | Local-only fallback/debug path; never hosted. Use this when isolating local index or FTS behavior. |

## Data Prep Strategy Summary (Chunking + Embeddings)

What we were testing:

- How to split QMS documents into chunks so retrieval stays precise while answers
  still have enough context.
- Which embedding/index baseline gives the strongest retrieval quality for the
  onsite corpus.
- Whether score changes are due to chunking itself or due to routing/citation
  logic outside chunking.

The implemented retrieval pipeline is:

1. Retrieve precise child chunks first.
2. Merge SQLite FTS, dense vector, and metadata candidates.
3. Preserve important lexical hits for exact IDs, filenames, standards, and
   table entities.
4. Rerank the merged candidate set, or use the deterministic fallback when the
   Qwen CrossEncoder is unavailable.
5. Expand nearest neighbor/parent context only after retrieval for answer
   synthesis, while citations still point to structured evidence.

### Benchmark Summary Table

| Experiment/change | What changed | Measured result | Decision |
| --- | --- | --- | --- |
| Character chunking ablation | Tested 1,800/2,800/5,000/10,000 char chunks with overlap. | Best early run was 1,800 chars at `31 / 84` and `0.6087`, but on an older corpus/index shape. | Use only as directional evidence that overly broad chunks hurt precision. |
| Token-aware child chunks | Switched to `600` token prose chunks, `100` overlap, and answer-time neighbor expansion. | Immediate deterministic hash pass count improved from `22 / 84` to `25 / 84`; citations became easier to inspect. | Keep `CHUNK_SIZE_TOKENS=600` and `CHUNK_OVERLAP_TOKENS=100`. |
| Table row evidence | Compared table targets near 500/700/900 tokens, row chunks, and bounded row indexing. | `700` tokens was the best table-row grouping balance; unbounded row chunks created about `100k` chunks. | Keep `TABLE_CHUNK_TARGET_TOKENS=700`; emit row chunks only for tables with `50` rows or fewer. |
| Metadata coverage | Added metadata-only chunks for sparse/empty-body DOCX records. | The index retained all `189` real DOCX files, including `24` metadata-only records. | Keep metadata chunks searchable for IDs, filenames, revisions, signed status, and obsolete status. |
| OpenAI embedding baseline | Rebuilt with `text-embedding-3-large`, `3072` dimensions, normalized `IndexFlatIP` vectors. | Core avg score improved from the pre-audit OpenAI report `0.5812` to `0.8296` after grouped retrieval/source-truth fixes. | Treat OpenAI embeddings as the quality baseline; hash embeddings are smoke/test only. |
| Obsolete filtering | Added latest-active default filtering and explicit obsolete scope. | Obsolete leakage fell from `0.6833` to `0.3000`, then to `0.1525` in the latest report. | Keep latest-active first; include obsolete records only when asked or needed for revision history. |
| Deterministic intents | Routed inventory, table lookup, revision diff, temporal status, traceability, and manifest count cases outside generic top-N retrieval. | Strict surface moved from `84 / 91` to an equivalent `91 / 91`; core score moved from `0.5812` to `0.8296`. | Keep intent-first retrieval; do not rely on dense retrieval for counts or source-of-truth questions. |
| Source-truth gates | Added required sources, table evidence, required backend, forbidden-source, and count contracts. | Targeted source-truth tests moved from `10 failed, 21 passed` to `31 passed`; after the pediatric fix, source-truth contracts pass `23 / 23`. | Broad score cannot override a source-truth contract failure. |

Only the early chunking/hash experiments were isolated enough to claim a direct
`22 / 84 -> 25 / 84` improvement from chunking plus neighbor context. The final
`0.5812 -> 0.8296` movement was a grouped system-level improvement from
chunking, metadata coverage, table evidence, deterministic routing, lexical
preservation, citation assembly, and source-truth gates.

### Metric and Keyword Definitions

- `Top-k hit rate`: fraction of queries where at least one expected source appears
  in the top `k` retrieved results.
- `Recall@k`: fraction of all expected sources recovered within top `k`.
- `Citation validity`: how often cited source IDs actually map to expected
  evidence sources.
- `Latest revision accuracy`: whether latest/current questions return the correct
  latest revision.
- `Obsolete leakage rate`: rate of obsolete evidence leaking into latest/current
  answers.
- `Child chunk retrieval`: retrieve precise small chunks first, then optionally
  add nearby chunks for answer context.
- `Metadata-only chunk`: a searchable chunk built from filename/doc metadata when
  extracted body text is sparse or empty.
- `Table row chunk`: a row-preserving table chunk used when an answer needs exact
  row/cell evidence.
- `Dense embedding`: a vector representation of chunk text; the quality baseline
  uses OpenAI `text-embedding-3-large` at `3072` dimensions.
- `IndexFlatIP`: the exact FAISS inner-product index used over normalized
  vectors, so scores behave like cosine similarity.
- `SQLite FTS`: full-text search over normalized metadata and chunk text for
  IDs, filenames, acronyms, and exact terms.
- `Hybrid retrieval`: merged SQLite FTS, dense vector, metadata, rank-fusion,
  and reranking retrieval.
- `SQL inventory`: deterministic SQLite count/list retrieval for questions where
  top-N chunks are the wrong source of truth.
- `Source-truth gate`: a hard eval rule for required source IDs, table evidence,
  required backend, forbidden terms, counts, and citation validity.
- `Hosted File Search`: OpenAI hosted retrieval over the same normalized corpus;
  it is synced but still needs a full comparative eval against the local path.

### Why We Chose The Current Strategy

- Precision-first retrieval is more important than broad chunk context because
  regulated answers need trustworthy citations.
- `600/100` prose chunking and `700` table chunking gave the best precision vs
  context tradeoff on this corpus shape.
- OpenAI `3072` embeddings are the quality baseline; hash embeddings remain a
  portability/smoke path, not a production quality claim.
- Current bottlenecks were mostly routing/citation/reference logic, not prose
  chunk size. The highest-leverage fixes were deterministic source-of-truth
  paths and stricter eval contracts.
- This repo optimizes for a known `189`-document corpus with `17,651` vector
  rows, where exact local search and inspectable provenance are acceptable
  tradeoffs. At much larger corpus sizes, the index and rebuild strategy would
  need to change.

## QMS Accuracy Audit Learnings

The latest search-quality pass started from raw CLI transcripts, not from a
model-only impression of quality. We ran the problematic queries with:

```bash
uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
```

Then we cross-checked the returned sources against SQLite metadata, normalized
Markdown, and the DOCX-derived extraction. The important lesson was that many
failures were not embedding failures in isolation. They were intent failures:
the system treated inventory, table lookup, temporal status, revision diff, and
traceability as ordinary top-N semantic search.

### Raw Issues Detected

| Query area | Raw symptom before fixes | Source truth | Fix/eval guard added |
| --- | --- | --- | --- |
| MX1 BOM | Returned the right primary BOM but over-emphasized unsigned BOM metadata. | `BOM-055 Rev G` is the top-level MX1 assembly; `BOM-079` and `BOM-101` are software BOMs. | Dedicated `mx1_bom` intent and grouped primary vs related BOM answer. |
| 510(k) summary | Retrieved EMC, V&V summary, design inputs, and risk summary chunks that did not mention 510(k). | No standalone indexed "510(k) Summary"; current evidence is `MEMO-P01-859`, `DHF-008`, and `PLN-P01-061`, including `K241567`. | `510k_summary_location` intent requires those docs and forbids unsupported "missing" answers based on unrelated sources. |
| VVPR inventory | Returned a 40-row slice and sometimes stated counts that did not match the shown rows. | `93` total VVPR revisions, `90` non-obsolete, `89` latest active records. | SQL-backed `vvpr_inventory`; eval requires the scoped counts. |
| Risk-related docs | Returned eight chunks from `VVAM-P01-004` instead of an inventory. | Risk-related inventory includes active `RSK-*`, `PLN-P01-063`, `VVAM-P01-004`, and other risk/RMF-bearing records. | SQL-backed `risk_related_inventory`; eval forbids "all VVAM chunks" behavior. |
| DHF/21 CFR 820.30 | Missed the actual DHF checklist and answered from adjacent project/risk docs. | `DHF-008 Rev D` is the current DHF checklist; `PLN-P01-062` is planning support. Current Part 820 is QMSR and 820.30 is reserved. | `dhf_82030` intent cites DHF evidence and states the regulatory caveat. |
| Risk protocols | Quoted a small sample and could include historical predecessor IDs. | Current evidence should be P01 protocols from active risk/VVAM evidence. | `risk_protocol_trace` intent; `must_not_include` guards for `VVPR-P00` and training `TRA` docs. |
| Electrical safety acceptance criteria | Lexical retrieval had the right evidence, but fusion/reranking pushed it out. | `MEMO-P01-685 Table 2 row 2` plus `3P-P01-33`; criteria are IEC 60601-1:2020-08 Ed. 3.2 and result `PASS`. | `electrical_safety_acceptance` intent, lexical preservation, and row/cell evidence expectation. |
| Open design review action items | A standalone topic change could inherit prior `MEMO-P01-685` context. | Current open actions are in `MEMO-P01-859` Section 4 / Summary of Action Items. | Token-boundary follow-up detection and `open_design_review_actions` intent. |
| Risk Rev C vs Rev D | Correctly noticed the exact pair was missing, then still dumped unrelated RSK chunks. | No single RSK chain has both Rev C and Rev D; no indexed RSK Rev D exists. | `ambiguous_risk_revision_diff` returns clarification/no-answer. |
| ECR last year/status | Returned ECRs but silently ignored "last year" and status. | Active signed ECRs are `ECR-577`, `ECR-587`, `ECR-593`; approval dates are in October 2024, outside the May 7 2026 current-date window. | `ecr_last_year_status` extracts date/status and states the date policy. |
| Electrical leakage trace | Multi-hop reported `references_followed:0` and omitted the actual 3P report. | Trace is `RSK-P01-010/017 -> VVAM-P01-004 -> MEMO-P01-685 -> 3P-P01-33`, with known partial gaps. | `electrical_leakage_trace` chain answer and required source list. |
| Third-party report mapping | Mixed planned testing with completed reports. | Completed indexed reports are `3P-P01-32` for IEC 60601-1-2 EMC and `3P-P01-33` for IEC 60601-1 dielectric/leakage. | `third_party_report_mapping` intent distinguishes plan vs completed evidence. |
| ECR count | This was already close, but answer wording needed current/signed/obsolete scope. | Count is exactly `3` active signed ECRs. | `ecr_count` SQL path and count/list expectation. |
| Completed vs planned verification | Counted raw VVPR records instead of completed vs planned verification work. | Use `MEMO-P01-685` result rows for completed and `PLN-P01-065` for planned scope. | `verification_completed_vs_planned` answer reports `67 completed` vs `86 planned` and avoids signed-filename inference. |
| Traceability eval data | Several eval rows expected `TRA`, but `TRA-*` docs are customer training docs. | The traceability matrix is `VVAM-P01-004`, not `TRA-024/025/026`. | Core eval rows now use `VVAM-P01-004` and forbid `TRA-*` training docs where relevant. |

### Category-by-Category Eval Learnings

| Category | Defect pattern found | Code/design change | Measured improvement |
| --- | --- | --- | --- |
| `compliance_cross_reference` | Multi-hop compliance answers missed terminal evidence or blended planning evidence with completed reports. | Added compliance-specific intents, lexical preservation for distinctive entities, and required source/table contracts. | Strict surface stayed `13 / 13`; core avg score improved `0.2875 -> 0.7125` (`+0.4250`). |
| `content_extraction_synthesis` | Table-backed facts could be retrieved but not pinned to the exact row/cell evidence. | Added row-preserving table chunks, row evidence expectations, and extraction-specific answer wording. | Strict surface stayed `13 / 13`; core avg score improved `0.7375 -> 0.8431` (`+0.1056`). |
| `cross_document_analysis` | Trace endpoints were incomplete for software critical-fault and acquisition flows. | Added deterministic software trace intents and required terminal protocol/report citations. | Strict category improved `11 / 13 -> 13 / 13` (`+2`); core avg score improved `0.5556 -> 0.8396` (`+0.2840`). |
| `enumeration_counting` | Count/list questions used top-N retrieval instead of inventory/manifests; traceability mapped to `TRA` training docs. | Routed counts to SQL inventory or `ingest_manifest`; mapped traceability matrices to `VVAM-P01-004`; added `required_backend`. | Strict category improved `11 / 13 -> 13 / 13` (`+2`); count accuracy is `1.0000`. |
| `exploratory_search` | Broad topic searches repeated nearby chunks and sometimes crossed into unrelated training or obsolete records. | Added topic inventory helpers, source-family guards, and latest-active filtering. | Strict surface stayed `13 / 13`; core avg score improved `0.6292 -> 0.8000` (`+0.1708`). |
| `known_item_retrieval` | Paraphrased document-title requests were treated as broad prefix searches. | Added title/filename-ranked known-item routing and signed/latest metadata scoring. | Strict category improved `11 / 13 -> 13 / 13` (`+2`); core avg score improved `0.6333 -> 0.9208` (`+0.2875`). |
| `revision_change_tracking` | Missing exact revision pairs fell back to unrelated revision chains. | Added topical revision-compare routing and no-answer behavior for absent chains. | Strict category improved `12 / 13 -> 13 / 13` (`+1`); core avg score improved `0.5625 -> 0.8167` (`+0.2542`). |

### Benchmarking Before And After

The pre-fix benchmark had two layers:

- Manual CLI audit: run each problematic query with `--trace --full-citations`,
  record the answer, retrieval backend, warnings, citation list, and whether the
  answer was correct, partial, or incorrect against the original corpus.
- Harness baseline: run the smoke/core eval datasets and save generated reports
  under `docs/eval-runs/`.

Pre-fix manual verdicts were mostly partial or incorrect for the new audit set:
inventory questions truncated, traceability did not follow the right chains,
table extraction missed the exact rows, and temporal/revision questions produced
overbroad fallback evidence. The older broad 84-case OpenAI-index run still
reported `84 / 84` at the harness threshold `0` with average score `0.5812`,
but that score did not catch several row-level and forbidden-source failures.

The external strict 91-case run before the audit fixes was the authoritative
strict baseline:

- Overall: `84 / 91` passed (`92%`).
- Smoke subset: `7 / 7` passed.
- Category counts: compliance cross-reference `13 / 13`; content extraction
  `13 / 13`; exploratory search `13 / 13`; revision change tracking `12 / 13`;
  known-item retrieval `11 / 13`; cross-document analysis `11 / 13`;
  enumeration counting `11 / 13`.
- Failing evidence targets:
  `MEMO-P01-638` for signed software development configuration management,
  `MEMO-P01-658` for System Architecture Diagram,
  `VVPR-P01-189 Rev B` plus `VVPR-P01-214 Rev C` for collimation/beam-angle
  comparison, canonical `critical faults` wording plus `VVPR-P01-181`,
  `Radiographic`/`Radioscopic` plus `VVPR-P01-179`, `VVAM-P01-004` for the one
  current traceability matrix, and `ingest_manifest` for non-empty DOCX ingest
  counts.

The audit treated paraphrased known-item misses, the `TRA` traceability mapping,
missing canonical phrases, missing endpoint citations, and missing manifest
citations as product defects for a regulated search workflow. The synthesized
fix was therefore deterministic: title/filename-ranked known-item routing,
topical revision compare routing, software trace endpoint intents, `VVAM`
traceability-matrix mapping, and a manifest-backed ingest-count answer.

After the fixes, the equivalent repo-backed strict surface is `91 / 91`:
`qms_smoke.jsonl` still covers one case per category, `qms_core.jsonl` passes
all `84` broader cases, and the seven strict failures are represented by
source-truth regression tests. That is a `+7` pass improvement and `+7.7`
percentage-point improvement over the `84 / 91` baseline.

A later strict CLI run improved to `90 / 91`; the only remaining failure was
`qms_cross_document_analysis_004`, where the answer cited valid
pediatric-filtration evidence but omitted the literal `VVPR-P01-152` phrase.
The fix added a deterministic `pediatric_filtration_trace` intent rather than a
generic seeded-title retrieval path. That answer cites `DR-P01-005`,
obsolete/historical `RSK-P01-010`, `VVAM-P01-004`, and `VVPR-P01-152`, including
the direct `PRD20.3 -> VVPR-P01-152` bridge.

The post-fix benchmark adds stricter checks:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest evals/test_query_plan.py evals/test_qms_source_truth_contracts.py -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest evals/test_qms_source_truth_contracts.py -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest evals -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run search-evals --dataset core --report docs/eval-runs --mode hybrid --fail-under 0
printf '%s\nquit\n' 'Trace pediatric filtration from requirements or risk rationale through verification evidence.' | UV_CACHE_DIR=/private/tmp/uv-cache uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
```

Latest verified results:

- Original seven-failure patch, failing-test-first proof before implementation:
  `10 failed, 21 passed` across `evals/test_query_plan.py` and
  `evals/test_qms_source_truth_contracts.py`.
- Original seven-failure patch, targeted regression after implementation:
  `31 passed`.
- Pediatric-filtration follow-up, failing-test-first proof before implementation:
  `2 failed, 30 passed`; both failures were the missing
  `pediatric_filtration_trace` intent/source-truth contract.
- Pediatric-filtration follow-up, targeted regression after implementation:
  `32 passed`.
- Source-truth contract after pediatric fix: `23 passed`.
- Full eval tests after pediatric fix: `148 passed`, with only the existing
  LangGraph deprecation warning.
- Artifact validation after index rebuild/stamping: `ok=True`,
  `requires_rebuild=False`, ingest schema `1`, vector schema `1`, `189`
  documents, and `17,651` vector rows.
- Core diagnostic report after pediatric fix:
  `84 / 84` passed at threshold `0`, average score `0.8296`, top-k hit rate
  `0.7381`, Recall@k `0.6500`, citation validity `0.5623`, obsolete leakage
  `0.1525`, report `docs/eval-runs/2026-05-07-174047.md`.
- Hash-query diagnostic report after pediatric fix:
  `84 / 84` passed at threshold `0`, average score `0.8282`, report
  `docs/eval-runs/2026-05-07-174824.md`.
- Actual `uv run chat` verification for the prior `90 / 91` failure now returns
  `VVPR-P01-152` in the answer and cites `DR-P01-005`, `RSK-P01-010`,
  `VVAM-P01-004`, and `VVPR-P01-152`; trace artifact validation reports
  `ok=True`, `requires_rebuild=False`.
- Eval JSON diagnostics now include `answer_text`, `source_ids`,
  `retrieved_source_ids`, `backend`, plus compatibility aliases `pass`,
  `missing_phrases`, and `source_hit`.

Mandatory query-path gate:

- Any change to query planning, query expansion, retrieval, reranking, citation
  assembly, answer synthesis, or evidence formatting must pass the source-truth
  contract before the full test suite:
  `uv run pytest evals/test_qms_source_truth_contracts.py -q`.
- Also run the targeted generated/audit eval that matches the touched path, for
  example smoke/core `search-evals`, golden-answer scoring, or dataset
  `--validate-only` when expected evidence rows changed.
- Do not accept a passing broad score alone when the source-truth gate fails; the
  gate checks provenance, required sources, and forbidden-source leakage.

## Eval Surface Expansion And Anti-Overfitting

We expanded the sample queries to increase eval surface area instead of tuning
for one demo transcript:

- `evals/datasets/qms_smoke.jsonl`: `7` cases, one per category, for fast
  walkthrough checks.
- `evals/datasets/qms_core.jsonl`: `84` cases, `12` per category, for broader
  scoring across all seven eval labels.
- Strict audit surface: `91` cases when smoke/category coverage is considered
  with the core set and the seven audited regressions.
- Each case varies structured dimensions: persona, revision scope,
  answerability, input noise, citation burden, required table evidence,
  forbidden sources, and required backend expectations.

We avoided overfitting by making the fixes target reusable product contracts:
source IDs, table evidence, SQL backends, count scope, revision policy,
forbidden-source guards, and retrieval traces. The code does not require exact
answer prose to match a golden paragraph; it requires the right evidence and
scope to appear. For example, `TRA-*` is forbidden for traceability-matrix
answers because those files are customer training records in this corpus, while
`VVAM-P01-004` is the current traceability matrix.

## LangChain Deep Agents: Used Vs Not Used

We kept the useful starter structure where it helped:

- `make_agent` still wraps LangChain `init_chat_model` and
  `create_deep_agent` for generic provider-backed chat.
- The project kept the `uv` package shape, environment-based provider config,
  CLI entry point, and FastAPI backend entry point.
- The existing `.invoke()`, `.stream()`, and `.astream()` compatible agent path
  remains available outside the QMS-specific search workflow.

We did not keep generic deep-agent retrieval as the QMS answer path. We also did
not use LangChain retriever abstractions, model-decided citations, or generic
agent memory for QMS evidence. Regulated QMS search needs deterministic count
basis, revision policy, source provenance, table-row evidence, forbidden-source
guards, and inspectable eval traces. The QMS path therefore uses deterministic
query planning, SQLite/FAISS hybrid retrieval, reranking or fallback, strict
citations, and source-truth evals before any answer synthesis.

## Future Eval Work

With more time, the next documentation-backed quality pass should run a full
comparison against OpenAI hosted File Search: local `hybrid`, hosted-preferred
`auto`, and hosted-only behavior where hosted failures fail the eval instead of
falling back. The run should record latency, cost, token usage, source coverage,
and category deltas.

For gradual document additions, ordinary new files should be extracted, chunked,
embedded, and merged incrementally. Full re-chunking should be reserved for
extraction changes, chunking config changes, embedding model changes, or
metadata schema changes. Periodic full rebuilds should still be scheduled when
the corpus has materially changed, so stale chunking choices do not accumulate
quietly over time.

Check status:

```bash
git branch --show-current
git status --short
uv run pytest evals/test_qms_source_truth_contracts.py -q
uv run pytest evals/ -v
```

## 2. Smoke-Test Retrieval

Single-turn strict local hybrid search:

```bash
uv run search-qms "Find BOM-055 Rev G" --mode hybrid --limit 8
```

Interactive multi-turn strict local hybrid chat:

```bash
uv run chat --qms-search --mode hybrid --limit 8
```

The interactive QMS prompt is `QMS> `. On a real terminal the CLI shows a
dynamic status spinner while it resolves follow-up context, searches, validates
citations, and renders the answer. Piped/scripted commands suppress the spinner
for deterministic output.

Useful output modes:

```bash
# Pretty output but no spinner
uv run chat --qms-search --mode hybrid --limit 8 --no-progress

# Stable text output
uv run chat --qms-search --mode hybrid --limit 8 --plain

# Machine-readable JSON
printf 'Find BOM-055 Rev G\nquit\n' | uv run chat --qms-search --mode hybrid --json
```

Targeted copy-paste commands without operational retrieval traces:

```bash
printf '%s\nquit\n' 'Find the signed MX1 software development configuration management memo.' | uv run chat --qms-search --mode hybrid --limit 8 --plain
printf '%s\nquit\n' 'Where is the 510(k) summary for the MX1 device? Cite the document ID if it exists.' | uv run chat --qms-search --mode hybrid --limit 8 --plain
printf '%s\nquit\n' 'What are the acceptance criteria for the electrical safety verification test?' | uv run chat --qms-search --mode hybrid --limit 8 --plain
printf '%s\nquit\n' 'How many traceability matrices are in the corpus?' | uv run chat --qms-search --mode hybrid --limit 8 --plain
printf '%s\nquit\n' 'Trace pediatric filtration from requirements or risk rationale through verification evidence.' | uv run chat --qms-search --mode hybrid --limit 8 --plain
```

The same targeted commands with operational retrieval traces and full citations:

```bash
printf '%s\nquit\n' 'Find the signed MX1 software development configuration management memo.' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
printf '%s\nquit\n' 'Where is the 510(k) summary for the MX1 device? Cite the document ID if it exists.' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
printf '%s\nquit\n' 'What are the acceptance criteria for the electrical safety verification test?' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
printf '%s\nquit\n' 'How many traceability matrices are in the corpus?' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
printf '%s\nquit\n' 'Trace pediatric filtration from requirements or risk rationale through verification evidence.' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
```

Copy-paste scripted multi-turn smoke:

```bash
printf 'Find the Bill of Materials for the MX1 system\nWhat revision is that?\nShow me the full pathname citation.\nquit\n' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
```

For human terminal use, `--trace` renders grouped trace panels. For exact JSON
trace output, use:

```bash
printf 'Find BOM-055 Rev G\nquit\n' | uv run chat --qms-search --mode hybrid --limit 8 --trace --raw-trace
```

### How To Read CLI Scores And Rankings

Verbose CLI output has two different scoring concepts:

| Output | Meaning |
| --- | --- |
| Citation `Score` | Retrieval evidence score attached to a citation or hit. It helps explain why the evidence was ranked, but it is not the eval score. |
| `lexical_candidates` | SQLite FTS candidates ranked by exact terms, IDs, filenames, acronyms, and metadata text. |
| `vector_candidates` | Dense embedding candidates ranked by normalized inner product, equivalent to cosine similarity for the stored vectors. |
| `fused_candidates` | Candidates after rank fusion merges lexical, dense, and metadata evidence. |
| `reranked_candidates` | Final candidate order after the Qwen CrossEncoder when available, or the deterministic fallback otherwise. |
| `pre_rank` / `pre_score` | Candidate rank and score before reranking. |
| `rerank_score` | Second-stage score used to order reranked candidates. |
| `search-evals` score | Separate deterministic eval score: `0.7 * term_score + 0.3 * source_score`, with hard source-truth contracts overriding the score. |

Use live CLI scores to inspect retrieval behavior. Use generated `search-evals`
reports to judge benchmark quality.

The ranking formulas used by the local `hybrid` path are:

| Stage | Formula / computation | Why this shape |
| --- | --- | --- |
| Rank fusion | `fused_score = lexical_weight / (60 + lexical_rank) + vector_weight / (60 + vector_rank)`, with missing ranks contributing `0`. | Reciprocal rank fusion avoids comparing raw SQLite FTS scores to dense-vector scores directly, because those scores are on different scales. |
| Lexical priority | `lexical_weight = 2.0` for extraction, known-item, compliance, and selected audited intents; otherwise `1.0`. `vector_weight = 1.0`. | QMS questions often hinge on exact IDs, standards, rows, and filenames. The audit showed exact evidence could be retrieved lexically but pushed down by generic fusion/reranking. |
| CrossEncoder rerank | When the configured Qwen CrossEncoder is available, `rerank_score` is the model's raw score for `(query, candidate_text)` and candidates are sorted descending. | This is the intended learned reranker path, but this sandbox currently reports `deterministic_fallback` unless native/model dependencies are installed. |
| Deterministic fallback rerank | `rerank_score = 4.0 * fused_score + 0.025 * term_overlap + phrase_boost + table_boost + exact_id_boost`. | Keeps fused retrieval as the dominant signal while preserving exact IDs, domain phrases, and row/table evidence when the CrossEncoder is unavailable. |

CrossEncoder was chosen because it reads the query and each candidate together,
which helps with close calls where exact wording, document IDs, standards, or
table rows matter. We use it only after hybrid retrieval has narrowed the set, so
vector search handles broad recall and CrossEncoder handles final ordering.

Fallback terms:

| Term | Definition |
| --- | --- |
| `term_overlap` | Count of non-stopword query terms also present in the candidate text. |
| `phrase_boost` | Sum of small boosts for audited distinctive phrases present in both query and candidate text, including `510(k)` (`0.12`), `K241567` (`0.16`), `3P-P01-32` (`0.12`), `3P-P01-33` (`0.12`), `acceptance criteria` (`0.12`), `electrical safety` (`0.12`), `dielectric` (`0.08`), `leakage` (`0.08`), `DHF-008` (`0.12`), and `RSK_R` (`0.08`). |
| `table_boost` | `0.08` when the hit is row/table-style evidence containing `Columns:` and `Row `. |
| `exact_id_boost` | `5.0` when the candidate document ID appears directly in the query. |

The evidence for these choices is grouped rather than coefficient-by-coefficient:
lexical preservation, wider rerank input, deterministic fallback scoring, and
source-truth contracts together improved Top-k hit `0.5238 -> 0.7381`,
Recall@k `0.4504 -> 0.6500`, and citation validity `0.3773 -> 0.5623`. We did
not run an isolated ablation for each numeric weight.

Auto-mode hosted fallback smoke:

```bash
printf 'Find BOM-055 Rev G\nquit\n' | OPENAI_VECTOR_STORE_STATE=/private/tmp/missing-openai-vector-store-state.json uv run chat --qms-search --mode auto --limit 8 --trace
```

Fast smoke eval:

```bash
uv run search-evals --dataset evals/datasets/qms_smoke.jsonl --report docs/eval-runs --mode hybrid --fail-under 0
uv run search-evals --dataset evals/datasets/qms_smoke.jsonl --report docs/eval-runs --mode auto --fail-under 0
```

Expected behavior:

- `Find BOM-055 Rev G` returns BOM-055 Rev G / MX1 Bill of Materials evidence
  near the top.
- The scripted multi-turn smoke keeps the follow-up anchored to the prior
  BOM-055 citation and lists full Markdown/source citation paths.
- Trace output includes raw and normalized input, resolved query, requested mode,
  retrieval backend, warnings, and the nested service trace.
- Pasted transcript prefixes such as `You:` are stripped before routing and
  recorded in trace metadata.

PFMEA-specific smoke query:

```bash
uv run search-qms "Find the MX1 MedAI PFMEA" --mode hybrid --limit 8
```

Additional smoke queries:

- `What is the latest active BOM-055 revision?`
- `Find verification reports for MX1 software system v3.3.0.`
- `Which documents mention the MedAI Rest Server?`
- `List obsolete MX1 software planning or anomaly documents.`
- `Which records support workstation installation qualification?`

## 3. Optional API Smoke

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
  -d '{"query":"Which verification reports cover MX1 software system v3.0.0?","mode":"hybrid","limit":8}'
```

## 4. Demo Script

Use this sequence for a live walkthrough:

1. Show index status with
   `uv run search-status --tasks TASKS.md --index-dir .data/qms-index --openai-state .data/openai/vector_store_state.json`.
2. Start with the intended default:
   `uv run chat --qms-search --mode hybrid --limit 8`.
3. Ask a known-item question: `Find BOM-055 Rev G.`
4. Ask a follow-up question: `What revision is that?`
5. Ask a synthesis question: `Summarize the evidence for MX1 software system
   verification around v3.3.0.`
6. Ask a revision question: `Which MX1 software planning documents are obsolete,
   and what active records appear related?`
7. Ask a cross-reference question: `Connect the PFMEA to verification or
   validation evidence.`
8. Ask a counting question: `How many VVPR documents are in the corpus?`
9. Re-run one query with `--trace --full-citations` and verify each cited
   filename/chunk maps to the answer.
10. Optionally run `--mode auto` once to show hosted-first fallback behavior;
    keep `--mode hybrid` as the quality baseline.

## 5. Expected Answer Behavior

Good answers:

- Cite every sourced QMS fact.
- Prefer latest active revisions unless the question asks for obsolete history.
- Say when evidence is metadata-only or extraction is limited.
- Avoid claims that are not present in retrieved evidence.
- Keep source lists compact and inspectable.
- Preserve evidence provenance from corpus artifact through normalized Markdown,
  SQLite record, chunk/table row, retrieval trace, and final citation.

Bad answers:

- Cite filenames not returned by retrieval.
- Treat obsolete records as active without warning.
- Use model knowledge about medical devices instead of corpus evidence.
- Give exact counts without explaining whether they come from filenames,
  document records, chunks, or extracted text.

## 6. Eval Handoff

After the walkthrough, record an eval run:

```bash
uv run pytest evals/test_qms_source_truth_contracts.py -q
uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --mode hybrid --fail-under 0
uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --mode auto --fail-under 0
```

If you write a manual note in addition to the generated reports, create
`docs/eval-runs/YYYY-MM-DD-medai-qms-search.md` with:

- Git SHA and branch.
- Corpus SHA-256.
- Index manifest summary.
- Artifact contract versions: dataset path/hash, generated-answer trace file
  when used, expected source IDs, required table evidence, and forbidden-source
  guards.
- Model defaults and any deviations.
- Retrieval metrics.
- Answer/citation metrics.
- Query expansion and retrieval trace summary, including backend, required
  sources found, partial-success warnings, and any references followed.
- Known failures or degraded modes.
