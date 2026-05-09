# Troubleshooting

Use this guide when implementation or eval work stalls. If the same failure
appears more than twice, follow the bug policy in
`docs/bugs/known_failures.md` before trying a third fix.

## Quick Status

```bash
git branch --show-current
git status --short
git diff --stat
uv run pytest evals/ -v
```

Scripted checks:

```bash
scripts/status.sh
scripts/check.sh
scripts/dev.sh
scripts/review_app.sh
```

`scripts/dev.sh` starts the backend and frontend together. If `.env` has no
`OPENAI_API_KEY`, it reports that OpenAI-backed paths are unavailable and enables
the local hash fallback for development. `scripts/review_app.sh` expects the
backend and frontend to already be running and reports missing index artifacts,
missing hosted OpenAI state, backend/frontend downtime, and local fallback
conditions.

For frontend work:

```bash
cd frontend
npm install
npm run build
```

## Baseline Confusion

Symptoms:

- A report describes the committed index as if it were the final OpenAI
  embedding build.
- Production readiness is inferred from the deterministic local hash eval.
- Playwright MCP/browser checks fail from command-line Chromium before page
  assertions run.

Fix:

- Treat the downloaded OpenAI vector bundle as the dense indexed baseline: `189`
  DOCX records, `24` metadata-only records, `17,651` chunks,
  `embedding_provider=openai`, `text-embedding-3-large`, and 3072 dimensions.
- Regenerate SQLite/FTS locally with `uv run ingest-qms`; do not expect
  `qms.sqlite` or normalized Markdown to be committed. Hosted OpenAI File Search
  state is local-only under `.data/openai/`.
- Treat the current demo as OpenAI-vector-indexed and SQLite-backed after local
  ingest. The optional Qwen CrossEncoder reranker path is implemented, but this
  sandbox reports the deterministic fallback until the native dependencies and
  model cache are provisioned. `gpt-5.5` answer synthesis is implemented with
  citation-label validation and deterministic fallback.
- In the Codex macOS sandbox, command-line Playwright can fail with Chromium
  `MachPortRendezvousServer ... Permission denied`. `scripts/check.sh` skips
  only that exact sandbox signature after Python tests, status, backend startup,
  and frontend build pass. Any other Playwright failure is still a hard failure.

## Corpus Not Found

Symptoms:

- Index command cannot find `Example_QMS_-_MedAI.zip`.
- Corpus hash is missing from the manifest.

Checks:

```bash
ls -lh Example_QMS_-_MedAI.zip
```

Fix:

- Place the corpus zip at the repo root.
- Do not commit the corpus unless the project owner explicitly changes that
  policy.

## Zip Contains Extra Files

Symptoms:

- Document count is higher than expected.
- Files beginning with `._` appear in logs.
- `__MACOSX` paths appear in search results.

Fix:

- Ignore directories, `__MACOSX/`, and `._*` resource fork entries.
- Add a deterministic zip scanner test.
- Expected real document count for the known corpus is 189 `.docx` files.

## DOCX Extraction Is Empty

Symptoms:

- Known document exists but has no chunks.
- Answer synthesis has no evidence even though filename retrieval works.

Fix:

- Treat the file as metadata-only and preserve the source record.
- Return metadata-only records for known-item queries.
- Do not use metadata-only records for body-text synthesis unless the metadata
  itself supports the answer.

## Binary Parser Error

Symptoms:

- Unicode decode errors.
- Corrupt text from `.docx`.
- Attempts to use `.text()` on a binary response or file.

Fix:

- Use binary readers for `.docx`.
- Never parse `.docx`, PDF, images, or spreadsheets through text-response APIs.
- Add file extension and MIME/type checks near ingestion.

## OpenAI Authentication Or Model Error

Symptoms:

- Embedding or answer calls fail with authentication or model errors.

Checks:

```bash
grep '^OPENAI_API_KEY=' .env | cut -d= -f2- | wc -c
```

Fix:

- Set `OPENAI_API_KEY` in `.env`.
- Do not `source .env`; URLs and metacharacters can break shell parsing.
- Confirm the implementation uses `text-embedding-3-large` with
  `dimensions=3072` for embeddings and `gpt-5.5` for answers.

## Hosted File Search Sync Stale Or Failed

Symptoms:

- `mode=hosted` falls back even though `.data/openai/vector_store_state.json`
  exists.
- `sync-openai-file-search` uploads unexpected source files.
- Search results have hosted matches but no clickable local citations.

Fix:

- Hosted state is usable only when `status` is `synced`, a `vector_store_id`
  exists, and the file signatures match the normalized Markdown directory.
- Sync `.data/qms-index/normalized/*.md`; do not upload raw `.docx` files.
- Keep `.data/openai/` gitignored because it contains local OpenAI file and
  vector-store IDs.
- If state is `failed`, inspect the `error` field, fix credentials/network/API
  issues, and rerun `uv run sync-openai-file-search --force` when you need a
  clean hosted vector store.

## FAISS Import Fails

Symptoms:

- `ModuleNotFoundError: faiss`
- Native wheel install errors.
- `/stats` or `search-status` reports `vector_backend=numpy_fallback`.

Fix:

- Confirm the selected dependency is compatible with Python 3.13 and the host
  architecture.
- Prefer `faiss-cpu` for the local MVP unless GPU support is explicitly needed.
- Install the optional local native group when the review machine should use the
  real native backend:
  `uv sync --group native-search`.
- The NumPy fallback is expected and supported when native FAISS is unavailable;
  do not treat it as data loss unless the user explicitly requires native FAISS.
- Record the exact package/version fix in `docs/bugs/known_failures.md` if it
  takes more than two attempts.

## Embedding Dimension Mismatch

Symptoms:

- FAISS load/search fails with shape mismatch.
- Manifest says a dimension other than `3072`.

Fix:

- Rebuild the index with `text-embedding-3-large` and `dimensions=3072`.
- Fail fast on incompatible manifests rather than silently searching.
- Delete only generated index artifacts, not source files or unrelated work.

## Hash Index Used In Production

Symptoms:

- Production mode starts against an index built with `--hash-embeddings`.
- Eval or status output does not clearly say whether vectors are hash or OpenAI
  embeddings.

Fix:

- Add or check manifest/status fields that distinguish `embedding_provider=hash`
  from `embedding_provider=openai`.
- Allow hash embeddings only for local demo, smoke tests, and deterministic
  review.
- Rebuild with `uv run build-qms-index` without `--hash-embeddings` before
  claiming the production baseline.

## Hosted File Search State Missing Or Stale

Symptoms:

- `search-status` reports no hosted state or a corpus hash that does not match
  the local manifest.
- Hosted mode cannot resolve uploaded file IDs back to local documents.

Fix:

- Run `uv run sync-openai-file-search` with `OPENAI_API_KEY` available in
  `.env`.
- Do not `source .env`; inspect individual variables with `grep` and `cut` if
  needed.
- Confirm `.data/openai/vector_store_state.json` reports `status=synced`, the
  current corpus hash, and `file_count=189`.

## Reranker Unavailable

Symptoms:

- `/stats` reports `backend=deterministic_fallback`.
- `/stats` includes `warning=real_reranker_unavailable`.
- `fallback_reason` mentions `sentence_transformers` or a missing local model.

Fix:

- Install optional native dependencies with
  `uv sync --group native-search` when the machine should run the real local
  CrossEncoder path.
- Ensure the configured model in `RERANKER_MODEL` is available in the local
  Hugging Face cache, or set `RERANKER_MODEL` to a local CrossEncoder model
  path.
- Return fused lexical+dense ranking as the supported degraded fallback when the
  model cannot load.
- Surface degraded mode in `/index/status`, CLI output, and eval run notes.
- Do not hide reranker failures during evals.

## Bad Known-Item Retrieval

Symptoms:

- Exact document ID query does not find the matching record.
- Filename query is outranked by semantically similar chunks.

Fix:

- Verify lexical indexing includes document IDs and exact filenames.
- Add deterministic boosts for exact ID and filename matches.
- Add a regression eval case before changing ranking weights.

## Obsolete Record Used As Current

Symptoms:

- Answer cites an obsolete document as active/current.
- Latest active revision is not preferred.

Fix:

- Check metadata parsing for `Obsolete`, revision suffixes, and `Rev X` patterns.
- Add a table-driven parser test for the filename.
- Ensure answer synthesis receives status metadata with every evidence object.

## Hallucinated Citation

Symptoms:

- Answer cites a source not present in retrieval output.
- Citation ID cannot be opened in the UI.

Fix:

- Build prompts from structured evidence IDs.
- Validate answer citations against retrieved evidence before returning.
- If validation fails, repair with the model or return a structured error.

## API Or Frontend Shape Error

Symptoms:

- Frontend crashes on missing citations or search results.
- API returns an unexpected object/list shape.

Fix:

- Guard list responses with `Array.isArray`.
- Guard object responses with `data && !data.error`.
- Return structured errors from POST and indexing handlers.
- Add loading, empty, error, and partial-success UI states.
