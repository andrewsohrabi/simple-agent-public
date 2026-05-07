from __future__ import annotations

import re
from dataclasses import dataclass, field


DOC_ID_PATTERN = re.compile(r"\b(?:[A-Z0-9]{2,5}-(?:P\d{2}|SWV)?-?\d{3}|BOM-\d{3}|ECR-\d{3}|ESF-\d{3})\b", re.IGNORECASE)
REV_PATTERN = re.compile(r"\bRev(?:ision)?\s+([A-Z])\b", re.IGNORECASE)
PREFIXES = {
    "BOM": ["bill of materials", "bom"],
    "VVPR": ["verification protocol", "verification report", "vvpr"],
    "VVAM": ["acceptance matrix", "vvam"],
    "RSK": ["risk", "hazard", "pfmea"],
    "ECR": ["engineering change request", "ecr", "change request"],
    "ESF": ["engineering summary", "esf"],
    "TRA": ["traceability", "trace matrix", "requirements trace"],
    "DHF": ["design history file", "dhf"],
    "DMR": ["device master record", "dmr"],
    "DR": ["design review", "action item"],
    "IFU": ["instructions for use", "ifu"],
    "QSR": ["quality system", "capa", "nonconformance"],
    "3P": ["third-party", "third party", "certification"],
}


@dataclass(frozen=True)
class QueryPlan:
    category: str
    strategy: str
    query: str
    doc_id: str | None = None
    prefix: str | None = None
    revision: str | None = None
    latest_only: bool = True
    include_obsolete: bool = False
    requires_count: bool = False
    requires_diff: bool = False
    warnings: list[str] = field(default_factory=list)


def plan_query(query: str) -> QueryPlan:
    q = query.strip()
    lower = q.lower()
    doc_match = DOC_ID_PATTERN.search(q)
    rev_match = REV_PATTERN.search(q)
    doc_id = doc_match.group(0).upper() if doc_match else None
    revision = rev_match.group(1).upper() if rev_match else None
    include_obsolete = "obsolete" in lower or "historical" in lower
    latest_only = not revision and "all revision" not in lower and "all versions" not in lower

    prefix = _infer_prefix(lower)
    if any(term in lower for term in ["how many", "count", "number of"]):
        return QueryPlan(
            category="enumeration",
            strategy="sql_count",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=False,
            include_obsolete=include_obsolete,
            requires_count=True,
        )
    if any(term in lower for term in ["changed between", "diff", "what changed", "compare rev"]):
        return QueryPlan(
            category="revision_change_tracking",
            strategy="revision_diff",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=False,
            include_obsolete=True,
            requires_diff=True,
        )
    if any(term in lower for term in ["trace", "map", "link", "through to", "cross-reference"]):
        return QueryPlan(
            category="cross_document_analysis",
            strategy="multi_hop",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=latest_only,
            include_obsolete=include_obsolete,
        )
    if any(term in lower for term in ["21 cfr", "fda", "required by", "compliance"]):
        return QueryPlan(
            category="compliance_cross_reference",
            strategy="hybrid",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=latest_only,
            include_obsolete=include_obsolete,
        )
    if any(term in lower for term in ["acceptance criteria", "summarize", "extract", "action items", "status"]):
        return QueryPlan(
            category="content_extraction",
            strategy="hybrid",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=latest_only,
            include_obsolete=include_obsolete,
        )
    if doc_id or any(term in lower for term in ["find", "where is", "show me the"]):
        return QueryPlan(
            category="known_item",
            strategy="exact_then_hybrid",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=latest_only,
            include_obsolete=include_obsolete,
        )
    return QueryPlan(
        category="exploratory",
        strategy="hybrid",
        query=q,
        doc_id=doc_id,
        prefix=prefix,
        revision=revision,
        latest_only=latest_only,
        include_obsolete=include_obsolete,
    )


def _infer_prefix(lower_query: str) -> str | None:
    explicit = {
        "ecr": "ECR",
        "engineering change request": "ECR",
        "vvpr": "VVPR",
        "verification protocol": "VVPR",
        "risk": "RSK",
        "traceability": "TRA",
        "bill of materials": "BOM",
        "bom": "BOM",
    }
    for term, prefix in explicit.items():
        if term in lower_query:
            return prefix
    for prefix, terms in PREFIXES.items():
        if any(term in lower_query for term in terms):
            return prefix
    return None
