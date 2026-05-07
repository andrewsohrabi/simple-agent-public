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

For frontend work:

```bash
cd frontend
npm install
npm run build
```

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

## FAISS Import Fails

Symptoms:

- `ModuleNotFoundError: faiss`
- Native wheel install errors.

Fix:

- Confirm the selected dependency is compatible with Python 3.13 and the host
  architecture.
- Prefer `faiss-cpu` for the local MVP unless GPU support is explicitly needed.
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

## Reranker Unavailable

Symptoms:

- Qwen reranker endpoint/model cannot load.
- Retrieval works but rerank step fails.

Fix:

- Return fused lexical+dense ranking as a degraded fallback.
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
