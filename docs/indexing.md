# Indexing Plan

This document defines the target indexing behavior for the MedAI QMS search MVP.

## Goals

- Build a repeatable local index from `Example_QMS_-_MedAI.zip`.
- Preserve enough metadata for strict citations and revision-aware search.
- Favor high-quality retrieval over premature scale optimizations.
- Make stale, incompatible, or degraded indexes obvious.

## Non-Goals

- Production document sync from SharePoint, Drive, or Documentum.
- OCR for scanned PDFs or images.
- Page-perfect citations from `.docx` files.
- Multi-tenant authorization.
- Managed vector database deployment.

## Source Handling

Index the QMS zip with binary-safe readers:

- Treat `.docx` as a binary Office file.
- Do not call `.text()` on `.docx`, PDF, image, or other binary artifacts.
- Ignore `__MACOSX`, `._*`, directories, and non-`.docx` files.
- Preserve the exact source filename for citations.
- Record extraction warnings per document.

Documents with little or no extractable body text should become metadata-only
records. They can satisfy known-item queries but should not support factual
synthesis unless their metadata is the evidence.

## Metadata Normalization

Parse and store:

- `doc_id`: examples include `VVPR-P01-179`, `MEMO-P01-638`, `BOM-055`.
- `family`: prefix before the first dash, such as `VVPR`, `MEMO`, `BOM`.
- `title`: normalized human title from the filename.
- `revision`: suffix revision such as `A`, `B`, `C`, or `Rev G`.
- `status`: `active`, `obsolete`, or `unknown`.
- `signed`: `true`, `false`, or `unknown`.
- `product_tokens`: `MX1`, `MedAI`, software version tokens, workstation IDs.
- `source_filename`: exact filename from the zip.

Revision precedence should be centralized and covered by tests. Active records
should rank ahead of obsolete records unless the query explicitly asks for
obsolete or historical records.

## Chunking

Chunking should balance citation quality and retrieval quality:

- Prefer section/heading boundaries when extraction exposes them.
- Include tables as text blocks with table context where possible.
- Target 600-token child chunks for prose.
- Use 100 tokens of overlap for long prose sections.
- Keep standalone prose chunks between 120 and 900 tokens where possible.
- Create one metadata chunk for every document.
- Split tables by row groups targeting 700 tokens with a 900-token maximum.
- Repeat compact table headers in every table chunk and never split a row across
  chunks.
- Expand selected child chunks with one neighboring chunk on each side for answer
  context when the parent section stays under 1,800 tokens.
- Keep chunk IDs stable across rebuilds for the same corpus and settings.
- Attach document metadata to every chunk.

Citation labels should include filename, revision/status, and chunk or section
ID. Page numbers are not required for `.docx` MVP citations unless a later
converter supplies stable page spans.

## Embeddings

Target production baseline:

- Model: `text-embedding-3-large`
- Dimensions: `3072`
- Index: FAISS `IndexFlatIP`

Build rules:

- Request embeddings with `dimensions=3072`.
- L2-normalize vectors before adding to FAISS.
- Store vector count and dimension in the manifest.
- Fail fast when loading an index whose stored dimension is not `3072`.
- Batch embedding requests and retry transient provider errors with backoff.

Current indexed baseline:

- The current local index uses OpenAI `text-embedding-3-large` embeddings at
  3072 dimensions, records `embedding_provider=openai`, and stores normalized
  vectors in the FAISS-compatible `IndexFlatIP` artifact.
- The deterministic hash path remains available through
  `build-qms-index --hash-embeddings` for smoke tests and offline regression
  work only.
- Hosted OpenAI File Search is synced for the same corpus hash and stores local
  state under `.data/openai/vector_store_state.json`.
- Query and document embeddings must continue to use the same
  provider/model/dimensions.

## Lexical Index

Dense embeddings alone are not enough for QMS content. Build a lexical index for:

- Exact document IDs.
- Filename/title terms.
- Revision labels.
- Acronyms and workstation IDs.
- Numeric software versions.

The implementation may use a simple BM25 library for the MVP. Persist enough
state to avoid rebuilding the lexical index on every app start.

## Candidate Fusion

Recommended retrieval flow:

1. Normalize query and detect exact IDs/revision/status hints.
2. Retrieve dense candidates from FAISS.
3. Retrieve lexical candidates from BM25 or equivalent.
4. Merge by chunk ID with score provenance.
5. Apply deterministic boosts for exact ID, filename, latest-active revision, and
   signed active records.
6. Rerank the top merged candidates with the Qwen reranker.
7. Return structured evidence objects.

If Qwen reranking is unavailable, return fused candidates, mark the response as
degraded, and include the degraded reason in status output.

## Manifest

Persist `manifest.json` next to the index files. Required fields:

- `schema_version`
- `created_at`
- `git_sha`
- `corpus_path`
- `corpus_sha256`
- `document_count`
- `skipped_count`
- `metadata_only_count`
- `chunk_count`
- `embedding_model`
- `embedding_dimensions`
- `vector_index_type`
- `vector_normalization`
- `lexical_index_type`
- `reranker`
- `chunking`
- `warnings`

The app should display manifest data in `GET /index/status` and the frontend
status panel.

Current SQLite status also includes first-class `revisions` and
`doc_references` tables. The latest verified status reports 189 documents,
7,778 chunks, 154 latest documents, 2,355 references, and a current schema.

## Validation

Minimum deterministic checks:

- Zip scanner returns 189 real `.docx` documents for the known corpus.
- Mac resource fork entries are ignored.
- Metadata parsing handles representative `BOM`, `MEMO`, `RSK`, `VVAM`, and
  `VVPR` filenames.
- Sparse or empty-body extraction creates metadata-only records.
- Embedding dimension is exactly `3072`.
- FAISS search returns stable top-k structure for a small fixture index.
- Latest-active revision outranks obsolete records for ambiguous queries.

Minimum live checks:

- Known-item query finds the expected source document in top 3.
- Cross-reference query returns at least two relevant document families.
- Revision query differentiates active and obsolete evidence.
- Counting query explains its counting basis.

Final verification note:

- Playwright MCP/browser verification has been run against the desktop
  workbench after production indexing and source-inspection behavior landed.
  Command-line Playwright remains a useful regression check, with the documented
  Codex macOS MachPort browser-launch failure treated as a sandbox-only skip by
  `scripts/check.sh`.
