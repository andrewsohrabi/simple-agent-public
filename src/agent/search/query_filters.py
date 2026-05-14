from __future__ import annotations

import re


_UNSIGNED_QUERY_RE = re.compile(
    r"\b(?:unsigned|non[-\s]?signed|not\s+signed)\b",
    re.IGNORECASE,
)
_SIGNED_QUERY_RE = re.compile(r"\bsigned\b", re.IGNORECASE)
_NON_OBSOLETE_QUERY_RE = re.compile(
    r"\b(?:non[-\s]?obsolete|not\s+(?:marked\s+)?obsolete)\b",
    re.IGNORECASE,
)
_OBSOLETE_QUERY_RE = re.compile(r"\bobsolete\b", re.IGNORECASE)


def requested_signature_filter(query: str) -> bool | None:
    """Return True for signed-only, False for unsigned-only, None for neither or mixed."""

    unsigned = bool(_UNSIGNED_QUERY_RE.search(query))
    signed_query = _UNSIGNED_QUERY_RE.sub(" ", query)
    signed = bool(_SIGNED_QUERY_RE.search(signed_query))
    if signed and unsigned:
        return None
    if signed:
        return True
    if unsigned:
        return False
    return None


def requested_obsolete_filter(query: str) -> bool | None:
    """Return True for obsolete-only, False for active-only, None for neither or mixed."""

    active_only = bool(_NON_OBSOLETE_QUERY_RE.search(query))
    obsolete_query = _NON_OBSOLETE_QUERY_RE.sub(" ", query)
    obsolete_only = bool(_OBSOLETE_QUERY_RE.search(obsolete_query))
    if active_only and obsolete_only:
        return None
    if obsolete_only:
        return True
    if active_only:
        return False
    return None


def document_scope_clauses(
    *,
    query: str,
    doc_id: str | None = None,
    prefix: str | None = None,
    revision: str | None = None,
    latest_only: bool = True,
    include_obsolete: bool = False,
) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    values: list[object] = []
    if doc_id:
        clauses.append("doc_id = ?")
        values.append(doc_id.upper())
    elif prefix:
        clauses.append("prefix = ?")
        values.append(prefix.upper())
    if revision:
        clauses.append("revision = ?")
        values.append(revision.upper())
    elif latest_only:
        clauses.append("is_latest = 1")

    obsolete_filter = requested_obsolete_filter(query)
    if obsolete_filter is True:
        clauses.append("is_obsolete = 1")
    elif obsolete_filter is False or not include_obsolete:
        clauses.append("is_obsolete = 0")

    signature_filter = requested_signature_filter(query)
    if signature_filter is not None:
        clauses.append("is_signed = ?")
        values.append(int(signature_filter))
    return clauses, values
