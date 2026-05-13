from __future__ import annotations

import re


_UNSIGNED_QUERY_RE = re.compile(
    r"\b(?:unsigned|non[-\s]?signed|not\s+signed)\b",
    re.IGNORECASE,
)
_SIGNED_QUERY_RE = re.compile(r"\bsigned\b", re.IGNORECASE)


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
