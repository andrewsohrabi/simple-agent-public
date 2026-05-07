from __future__ import annotations

from agent.search.schema import Citation, SearchHit
from agent.search.sqlite_store import SearchStore


def citations_for_hits(store: SearchStore, hits: list[SearchHit]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[tuple[str, str, str]] = set()
    for hit in hits:
        citation = store.citation_for_chunk(hit.chunk_id)
        if citation is None:
            continue
        key = (
            citation.doc_id,
            citation.revision,
            citation.section,
            citation.chunk_id or "",
        )
        if key not in seen:
            citations.append(citation)
            seen.add(key)
    return citations


def validate_citations(store: SearchStore, citations: list[Citation]) -> list[str]:
    errors: list[str] = []
    for citation in citations:
        if not citation.doc_id:
            errors.append("Citation is missing doc_id")
        if not citation.revision:
            errors.append(f"Citation for {citation.doc_id or 'unknown'} is missing revision")
        if not citation.section:
            errors.append(
                f"Citation for {citation.doc_id or 'unknown'} Rev {citation.revision or 'unknown'} is missing section"
            )
        if citation.chunk_id:
            stored = store.citation_for_chunk(citation.chunk_id)
            if stored is None:
                errors.append(f"Unknown citation chunk_id {citation.chunk_id}")
                continue
            if (
                stored.doc_id != citation.doc_id
                or stored.revision != citation.revision
                or stored.section != citation.section
            ):
                errors.append(
                    "Citation chunk metadata mismatch "
                    f"{citation.chunk_id}: expected {stored.doc_id} Rev {stored.revision} "
                    f"{stored.section}, got {citation.doc_id} Rev {citation.revision} {citation.section}"
                )
        elif not _document_exists(store, citation.doc_id, citation.revision):
            errors.append(
                f"Unknown citation document {citation.doc_id} Rev {citation.revision}"
            )
    return errors


def validate_citation_rows(store: SearchStore, citations: list[dict[str, object]]) -> list[str]:
    rows: list[Citation] = []
    for citation in citations:
        rows.append(
            Citation(
                doc_id=str(citation.get("doc_id") or ""),
                revision=str(citation.get("revision") or ""),
                title=str(citation.get("title") or ""),
                section=str(citation.get("section") or ""),
                filename=str(citation.get("filename") or ""),
                markdown_path=(
                    str(citation.get("markdown_path"))
                    if citation.get("markdown_path") is not None
                    else None
                ),
                markdown_path_abs=(
                    str(citation.get("markdown_path_abs"))
                    if citation.get("markdown_path_abs") is not None
                    else None
                ),
                source_path=(
                    str(citation.get("source_path"))
                    if citation.get("source_path") is not None
                    else None
                ),
                source_path_abs=(
                    str(citation.get("source_path_abs"))
                    if citation.get("source_path_abs") is not None
                    else None
                ),
                chunk_id=(
                    str(citation.get("chunk_id"))
                    if citation.get("chunk_id") is not None
                    else None
                ),
                evidence_type=str(citation.get("evidence_type") or "metadata"),
                support_level=str(citation.get("support_level") or "document"),
                heading_path=tuple(
                    str(item)
                    for item in citation.get("heading_path", ())
                    if item
                )
                if isinstance(citation.get("heading_path", ()), (list, tuple))
                else (),
                table_index=_optional_int(citation.get("table_index")),
                row_start=_optional_int(citation.get("row_start")),
                row_end=_optional_int(citation.get("row_end")),
                columns=tuple(
                    str(item)
                    for item in citation.get("columns", ())
                    if item
                )
                if isinstance(citation.get("columns", ()), (list, tuple))
                else (),
                row_cells={
                    str(key): str(value)
                    for key, value in (citation.get("row_cells") or {}).items()
                }
                if isinstance(citation.get("row_cells"), dict)
                else {},
            )
        )
    return validate_citations(store, rows)


def _document_exists(store: SearchStore, doc_id: str, revision: str) -> bool:
    if not doc_id or not revision:
        return False
    with store.connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE doc_id = ? AND revision = ?",
            (doc_id.upper(), revision.upper()),
        ).fetchone()
    return row is not None


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
