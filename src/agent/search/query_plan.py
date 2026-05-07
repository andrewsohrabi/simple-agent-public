from __future__ import annotations

import re
from dataclasses import dataclass, field


DOC_ID_PATTERN = re.compile(r"\b(?:[A-Z0-9]{2,5}-(?:P\d{2}|SWV)?-?\d{3}|BOM-\d{3}|ECR-\d{3}|ESF-\d{3})\b", re.IGNORECASE)
REV_PATTERN = re.compile(r"\bRev(?:ision)?\s+([A-Z])\b", re.IGNORECASE)
REV_PAIR_PATTERN = re.compile(
    r"\bRev(?:ision)?\s+([A-Z])\b.*?\bRev(?:ision)?\s+([A-Z])\b",
    re.IGNORECASE,
)
PREFIXES = {
    "BOM": ["bill of materials", "bom"],
    "PLN": ["planning", "plan", "project quality plan", "pln"],
    "MEMO": ["memo", "memorandum"],
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
    requires_list: bool = False
    requires_revision_chain: bool = False
    requires_diff: bool = False
    compared_revisions: tuple[str, str] | None = None
    warnings: list[str] = field(default_factory=list)


def plan_query(query: str) -> QueryPlan:
    q = query.strip()
    lower = q.lower()
    doc_match = DOC_ID_PATTERN.search(q)
    rev_match = REV_PATTERN.search(q)
    rev_pair_match = REV_PAIR_PATTERN.search(q)
    doc_id = doc_match.group(0).upper() if doc_match else None
    revision = rev_match.group(1).upper() if rev_match else None
    compared_revisions = (
        (rev_pair_match.group(1).upper(), rev_pair_match.group(2).upper())
        if rev_pair_match
        else None
    )
    include_obsolete = any(
        term in lower
        for term in ["obsolete", "historical", "older revision", "all revision", "all version"]
    )
    if "revisions" in lower or "revision chains" in lower:
        include_obsolete = True
    asks_all = any(
        term in lower
        for term in [
            "all ",
            "list ",
            "show all",
            "survey",
            "available",
            "present",
            "revision history",
            "older revisions",
        ]
    )
    latest_only = not revision and not include_obsolete and not asks_all

    prefix = _infer_prefix(lower)
    if "risk file" in lower or "risk analysis" in lower:
        prefix = "RSK"
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
    if any(
        term in lower
        for term in [
            "revision history",
            "older revisions",
            "all revision chains",
            "all revisions",
            "which older revisions",
        ]
    ):
        return QueryPlan(
            category="revision_diff",
            strategy="revision_chain",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=False,
            include_obsolete=True,
            requires_list=True,
            requires_revision_chain=True,
        )
    if any(
        term in lower
        for term in [
            "changed between",
            "diff",
            "what changed",
            "compare rev",
            "compare the obsolete",
            "compare rev",
        ]
    ):
        return QueryPlan(
            category="revision_diff",
            strategy="revision_diff",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=False,
            include_obsolete=True,
            requires_diff=True,
            compared_revisions=compared_revisions,
        )
    if any(
        term in lower
        for term in [
            "filed in",
            "their status",
            "marked obsolete",
            "signed versus unsigned",
            "supports the transition",
        ]
    ):
        return QueryPlan(
            category="revision_diff",
            strategy="sql_list" if prefix else "hybrid",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=False,
            include_obsolete=include_obsolete,
            requires_list=bool(prefix),
        )
    if any(term in lower for term in ["trace", "map", "link", "through to", "cross-reference"]):
        return QueryPlan(
            category="traceability",
            strategy="multi_hop",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=latest_only,
            include_obsolete=include_obsolete,
        )
    if any(
        term in lower
        for term in [
            "21 cfr",
            "fda",
            "required by",
            "compliance",
            "auditor",
            "regulatory",
            "design history file include",
        ]
    ):
        return QueryPlan(
            category="compliance",
            strategy="hybrid",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=latest_only,
            include_obsolete=include_obsolete,
        )
    if any(
        term in lower
        for term in ["acceptance criteria", "summarize", "extract", "action items", "status"]
    ):
        return QueryPlan(
            category="extraction",
            strategy="hybrid",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=latest_only,
            include_obsolete=include_obsolete,
        )
    if prefix and any(
        term in lower
        for term in [
            "list",
            "show all",
            "show records",
            "show documents",
            "what records",
            "what documents",
            "what verification",
            "which records",
            "which verification",
            "survey",
        ]
    ):
        return QueryPlan(
            category="exploratory",
            strategy="sql_list",
            query=q,
            doc_id=doc_id,
            prefix=prefix,
            revision=revision,
            latest_only=latest_only,
            include_obsolete=include_obsolete,
            requires_list=True,
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
        "engineering change record": "ECR",
        "vvpr": "VVPR",
        "verification protocol": "VVPR",
        "verification protocols": "VVPR",
        "verification test protocol": "VVPR",
        "verification test protocols": "VVPR",
        "verification test": "VVPR",
        "verification report": "VVPR",
        "risk": "RSK",
        "risk-management": "RSK",
        "risk management": "RSK",
        "traceability": "TRA",
        "traceability matrices": "TRA",
        "bill of materials": "BOM",
        "bom": "BOM",
        "planning documents": "PLN",
        "project quality plan": "PLN",
        "instructions for use": "IFU",
        "ifu": "IFU",
        "third-party": "3P",
        "third party": "3P",
        "design history file": "DHF",
        "dhf": "DHF",
        "device master record": "DMR",
        "dmr": "DMR",
        "quality system": "QSR",
        "capa": "QSR",
    }
    for term, prefix in explicit.items():
        if term in lower_query:
            return prefix
    for prefix, terms in PREFIXES.items():
        if any(term in lower_query for term in terms):
            return prefix
    return None
