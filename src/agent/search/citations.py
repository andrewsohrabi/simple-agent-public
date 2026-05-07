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
        if citation.chunk_id and store.citation_for_chunk(citation.chunk_id) is None:
            errors.append(f"Unknown citation chunk_id {citation.chunk_id}")
    return errors
