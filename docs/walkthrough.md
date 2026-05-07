# MedAI QMS Demo Walkthrough

This walkthrough reflects the current OpenAI-backed indexed baseline and calls
out the remaining production gaps separately.

## Prerequisites

- Branch: `andrewsohrabi/valkai-onsite`
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

Retrieval modes for the walkthrough:

| Mode | Use in the demo |
| --- | --- |
| `auto` | Hosted OpenAI File Search first, then local fallback when hosted state is missing, stale, empty, or errors. |
| `hosted` | Hosted-preferred validation path with local fallback warnings. |
| `hybrid` | Strict local SQLite FTS + FAISS + reranker path; never hosted. Use this for the local quality baseline. |
| `local` | Local-only fallback/debug path; never hosted. Use this when isolating local index or FTS behavior. |

## Data Prep Strategy Summary (Chunking + Embeddings)

What we were testing:

- How to split QMS documents into chunks so retrieval stays precise while answers
  still have enough context.
- Which embedding/index baseline gives the strongest retrieval quality for the
  onsite corpus.
- Whether score changes are due to chunking itself or due to routing/citation
  logic outside chunking.

### Benchmark Summary Table

| Experiment | What changed | Why it was tested | Key result | Decision |
| --- | --- | --- | --- | --- |
| Character chunking ablation | Tested 1,800/2,800/5,000/10,000 char chunks with overlap | Quick directional signal on chunk granularity | Best early run: 1,800 chars at `31/84` and `0.6087`, but on older corpus/index shape | Keep only as directional evidence, not final baseline |
| Token-aware child chunk baseline | Switched to `600` token prose chunks with `100` overlap and answer-time neighbor expansion | Better citation precision with enough local context | Stable retrieval behavior and cleaner citation boundaries; local hash run improved immediate pre-token pass count (`22/84` -> `25/84`) | Chosen prose baseline: `CHUNK_SIZE_TOKENS=600`, `CHUNK_OVERLAP_TOKENS=100` |
| Table chunking matrix | Compared table targets around 500/700/900 tokens, row-level chunks, and bounded row indexing | Tables dominate this corpus; row-level evidence quality is critical | `700` remained the best row-group target; unbounded row chunks created ~100k chunks, so row chunks are emitted only for tables with 50 rows or fewer | Chosen table baseline: `TABLE_CHUNK_TARGET_TOKENS=700`, repeat compact headers, preserve table rows for audit-sized tables |
| Metadata coverage fix | Included sparse/empty-body docs as metadata-only chunks | Known-item and filename retrieval were missing important records | Index now retains all real DOCX records (`189` docs, `24` metadata-only) with `17,651` chunks after bounded row-level table indexing | Keep metadata chunks enabled and searchable |
| Embedding/index baseline | OpenAI `text-embedding-3-large` at `3072` dims with normalized `IndexFlatIP` vectors | Prioritize quality over cost for baseline | OpenAI-index evals are stronger and representative than hash-only smoke runs; latest local core report average score `0.5812` | Final baseline: OpenAI embeddings (`3072`) + `IndexFlatIP` |

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

### Why We Chose The Current Strategy

- Precision-first retrieval is more important than broad chunk context because
  regulated answers need trustworthy citations.
- `600/100` prose chunking and `700` table chunking gave the best precision vs
  context tradeoff on this corpus shape.
- OpenAI `3072` embeddings are the quality baseline; hash embeddings remain a
  portability/smoke path, not a production quality claim.
- Current bottlenecks are mostly routing/citation/reference logic, so further
  prose chunk-size tuning is lower leverage than deterministic retrieval fixes.

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
| Open design review action items | A standalone topic change could inherit prior `MEMO-P01-685` context. | Current open actions are in `MEMO-P01-859 Table 5`. | Token-boundary follow-up detection and `open_design_review_actions` intent. |
| Risk Rev C vs Rev D | Correctly noticed the exact pair was missing, then still dumped unrelated RSK chunks. | No single RSK chain has both Rev C and Rev D; no indexed RSK Rev D exists. | `ambiguous_risk_revision_diff` returns clarification/no-answer. |
| ECR last year/status | Returned ECRs but silently ignored "last year" and status. | Active signed ECRs are `ECR-577`, `ECR-587`, `ECR-593`; approval dates are in October 2024, outside the May 7 2026 current-date window. | `ecr_last_year_status` extracts date/status and states the date policy. |
| Electrical leakage trace | Multi-hop reported `references_followed:0` and omitted the actual 3P report. | Trace is `RSK-P01-010/017 -> VVAM-P01-004 -> MEMO-P01-685 -> 3P-P01-33`, with known partial gaps. | `electrical_leakage_trace` chain answer and required source list. |
| Third-party report mapping | Mixed planned testing with completed reports. | Completed indexed reports are `3P-P01-32` for IEC 60601-1-2 EMC and `3P-P01-33` for IEC 60601-1 dielectric/leakage. | `third_party_report_mapping` intent distinguishes plan vs completed evidence. |
| ECR count | This was already close, but answer wording needed current/signed/obsolete scope. | Count is exactly `3` active signed ECRs. | `ecr_count` SQL path and count/list expectation. |
| Completed vs planned verification | Counted raw VVPR records instead of completed vs planned verification work. | Use `MEMO-P01-685` result rows for completed and `PLN-P01-065` for planned scope. | `verification_completed_vs_planned` answer reports `67 completed` vs `86 planned` and avoids signed-filename inference. |
| Traceability eval data | Several eval rows expected `TRA`, but `TRA-*` docs are customer training docs. | The traceability matrix is `VVAM-P01-004`, not `TRA-024/025/026`. | Core eval rows now use `VVAM-P01-004` and forbid `TRA-*` training docs where relevant. |

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

The post-fix benchmark adds stricter checks:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest evals/test_qms_source_truth_contracts.py -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run python evals/run_search_evals.py --dataset smoke --mode hybrid --fail-under 0.95
UV_CACHE_DIR=/private/tmp/uv-cache uv run python evals/run_search_evals.py --dataset core --validate-only
```

Latest verified results:

- Full tests: `108 passed`, with only the existing LangGraph deprecation warning.
- 14-query source-truth contract: `14 passed`.
- Smoke eval: `7 / 7 passed`, average score `1.0`.
- Core dataset validation: `84` cases, `12` per category, schema valid.
- Eval data audit: no `TRA` or `TRA-*` values remain in `source_ids`; remaining
  `TRA` mentions are forbidden-source guards where appropriate.

Mandatory query-path gate:

- Any change to query planning, query expansion, retrieval, reranking, citation
  assembly, answer synthesis, or evidence formatting must pass the 14-query
  source-truth contract before the full test suite:
  `uv run pytest evals/test_qms_source_truth_contracts.py -q`.
- Also run the targeted generated/audit eval that matches the touched path, for
  example smoke/core `search-evals`, golden-answer scoring, or dataset
  `--validate-only` when expected evidence rows changed.
- Do not accept a passing broad score alone when the source-truth gate fails; the
  gate checks provenance, required sources, and forbidden-source leakage.

Check status:

```bash
git branch --show-current
git status --short
uv run pytest evals/test_qms_source_truth_contracts.py -q
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

Copy-paste scripted multi-turn smoke:

```bash
printf 'Find the Bill of Materials for the MX1 system\nWhat revision is that?\nShow me the full pathname citation.\nquit\n' | uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
```

For human terminal use, `--trace` renders grouped trace panels. For exact JSON
trace output, use:

```bash
printf 'Find BOM-055 Rev G\nquit\n' | uv run chat --qms-search --mode hybrid --limit 8 --trace --raw-trace
```

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
  -d '{"query":"Which verification reports cover MX1 software system v3.0.0?","mode":"hybrid","limit":8}'
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
2. Start in `auto` to show hosted-first behavior, then switch to `hybrid` to
   show strict local retrieval when needed.
3. Ask a known-item question: `Find BOM-055 Rev G.`
4. Ask a follow-up question: `What revision is that?`
5. Ask a synthesis question: `Summarize the evidence for MX1 software system
   verification around v3.3.0.`
6. Ask a revision question: `Which MX1 software planning documents are obsolete,
   and what active records appear related?`
7. Ask a cross-reference question: `Connect the PFMEA to verification or
   validation evidence.`
8. Ask a counting question: `How many VVPR documents are in the corpus?`
9. Open citations and verify each cited filename/chunk maps to the answer.

## 6. Expected Answer Behavior

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

## 7. Eval Handoff

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

## 8. Final Browser Verification

Playwright MCP/browser verification was run on the desktop workbench after the
OpenAI index, hosted/local retrieval modes, citation flow, and frontend source
inspection workflow were in place. The MCP check loaded
`http://127.0.0.1:3060`, ran the engineering-change-request enumeration example,
opened Sources and Debug, and reported zero browser console warnings/errors.

Command-line Playwright remains in the repo for normal environments. In this
Codex macOS sandbox, `scripts/check.sh` skips only the documented Chromium
MachPort permission failure after backend startup and frontend build pass.
