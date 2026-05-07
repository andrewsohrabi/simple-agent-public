# Evaluation Plan

The search MVP needs both deterministic tests and live evals. Deterministic tests
protect parsing/indexing behavior. Live evals measure retrieval and answer
quality with real models.

## TDD Workflow

For each implementation phase:

1. Add the smallest failing test that describes the desired behavior.
2. Implement only enough code to pass.
3. Run the targeted tests.
4. Run `uv run pytest evals/ -v` before handing off a phase.
5. Record live eval outcomes in `docs/eval-runs/` when model calls or retrieval
   quality materially change.

Do not wait until the UI is complete to add retrieval evals. Retrieval quality is
the foundation for answer quality.

## Eval Sets

Create eval cases across these categories:

| Category | Purpose | Example |
| --- | --- | --- |
| Known item | Exact document/title retrieval | `Find the MX1 MedAI PFMEA.` |
| Revision | Latest active vs obsolete behavior | `What is the latest active BOM-055 revision?` |
| Cross-reference | Evidence across document families | `Connect PFMEA evidence to verification reports.` |
| Extraction | Pull a specific fact from a record | `What does the use specification say about MX1 use?` |
| Synthesis | Summarize multiple retrieved records | `Summarize software v3.3.0 verification evidence.` |
| Counting | Count documents or mentions with caveats | `How many VVPR files are in the corpus?` |
| Negative | Abstain when evidence is missing | `What does the corpus say about a product not present?` |

Each case should define:

- Query.
- Expected document IDs or families.
- Expected citations or citation constraints.
- Expected answer facts where applicable.
- Allowed caveats.
- Whether obsolete records are allowed.

The repo currently includes:

- `evals/datasets/qms_smoke.jsonl`: one prompt per category for fast harness checks.
- `evals/datasets/qms_core.jsonl`: 84 prompt cases, 12 per category.
- `evals/datasets/qms_smoke_golden_answers.jsonl`: deterministic answer and retrieval traces for the smoke set. Use this when retrieval is incomplete or when validating metric plumbing without model calls.

## Metrics

Retrieval metrics:

- Recall@3, Recall@5, Recall@10.
- MRR for known-item queries.
- Family coverage for cross-reference queries.
- Latest-active accuracy for revision queries.
- Metadata-only handling accuracy.

Implemented deterministic harness metrics:

- Top-k hit rate: at least one expected source appears in the ranked trace within `k` (default 5).
- Recall@k: expected source coverage within the ranked trace.
- Count accuracy: reported count equals `expected_count` when a golden trace supplies one.
- Citation validity: cited source IDs map to expected source IDs or prefixes.
- Latest revision accuracy: latest/current cases report the expected `Rev X`.
- Obsolete leakage rate: current/latest cases do not leak obsolete text or obsolete trace metadata.

Answer metrics:

- Correctness against expected facts.
- Citation coverage for all sourced claims.
- Citation precision: cited sources must support the claim.
- Abstention quality.
- Obsolete/current distinction.
- Count basis clarity.

Operational metrics:

- Index build time.
- Embedding token/request usage.
- Retrieval latency before and after reranking.
- Answer latency.
- Degraded-mode frequency.

## Baseline Run Configuration

Default run configuration:

```text
answer_model=gpt-5.5
embedding_model=text-embedding-3-large
embedding_dimensions=3072
vector_index=faiss.IndexFlatIP
reranker=qwen
retrieval=hybrid_lexical_dense
revision_policy=latest_active_first
```

Any deviation must be recorded in the eval run note.

## Eval Run Notes

Store run notes in `docs/eval-runs/YYYY-MM-DD-<summary>.md`.

The CLI can generate timestamped Markdown and JSON reports:

```bash
uv run search-evals --dataset core --report docs/eval-runs --mode local --fail-under 0
```

Deterministic smoke guardrail:

```bash
uv run search-evals --dataset smoke --answers-jsonl evals/datasets/qms_smoke_golden_answers.jsonl
```

In the Codex sandbox, use `UV_CACHE_DIR=/private/tmp/uv-cache` if uv cannot open the default user cache.

Required fields:

```md
# Eval Run: <summary>

- Date:
- Branch:
- Git SHA:
- Corpus SHA-256:
- Index manifest:
- Index manifest SHA-256:
- Model config:
- Answer model:
- Embedding model:
- Embedding dimensions:
- Vector index:
- Reranker:
- Retrieval settings:
- Degraded mode:

## Results

| Metric | Value | Notes |
| --- | ---: | --- |

## Category Pass Summary

| Category | Cases | Passed | Pass Rate | Avg Score |
| --- | ---: | ---: | ---: | ---: |

## Failures

## Follow-Ups
```

## Pass Criteria For MVP

Minimum target before demo:

- Known-item Recall@5: 90% or better on the curated eval set.
- Revision cases: no active/obsolete mislabeling in top cited answer.
- Citation coverage: every answer with corpus facts includes citations.
- Negative cases: answers abstain or ask for clarification instead of inventing.
- UI demo: index status and citations are visible without devtools.

These thresholds are starting points. If the eval set is very small, include raw
case results instead of relying on aggregate metrics alone.

## Latest OpenAI-Index Run

The latest full core run is
`docs/eval-runs/2026-05-07-031242.md`.

| Metric | Value |
| --- | ---: |
| Cases | 84 |
| Average score | 0.5639 |
| Top-k hit rate | 0.5238 |
| Recall@k | 0.4385 |
| Citation validity | 0.3815 |
| Latest revision accuracy | 1.0000 |
| Obsolete leakage rate | 0.6833 |

Interpretation:

- The harness ran all seven exercise categories against the real local OpenAI
  embedding index.
- The current system is usable for demo navigation and source-backed retrieval,
  but the eval report identifies the next quality work: citation precision,
  cross-document/reference recall, obsolete filtering, and real reranking.
- The run used `--fail-under 0` deliberately so the report captures every case
  instead of treating the current quality score as a production pass threshold.

## Failure Handling

When an eval fails:

1. Classify it as extraction, metadata, chunking, retrieval, reranking, synthesis,
   UI, or infrastructure.
2. Add a failing deterministic test when the issue is reproducible without live
   model calls.
3. If the same failure repeats more than twice, follow
   `docs/bugs/known_failures.md` before a third fix attempt.
4. Record unresolved eval failures in the run note and, when actionable, in
   `docs/bugs/bugs.md`.
