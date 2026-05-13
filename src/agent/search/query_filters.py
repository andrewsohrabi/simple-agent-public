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
