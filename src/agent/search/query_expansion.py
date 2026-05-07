from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AliasGroup:
    name: str
    terms: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class QueryExpansion:
    original_query: str
    normalized_query: str
    expanded_query: str
    terms: tuple[str, ...]
    matched_groups: tuple[str, ...]


def _compiled(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


ALIAS_GROUPS: tuple[AliasGroup, ...] = (
    AliasGroup(
        name="510k",
        terms=(
            "510(k)",
            "510k",
            "510 k",
            "K number",
            "premarket notification",
            "FDA 510(k)",
            "substantial equivalence",
        ),
        patterns=_compiled(
            r"\b510(?:\s*\(\s*k\s*\)|\s*k\b)",
            r"\bk\s*[- ]?\d{6}\b",
            r"\bk\s*[- ]?number\b",
            r"\bpremarket notification\b",
            r"\bsubstantial equivalence\b",
        ),
    ),
    AliasGroup(
        name="dhf",
        terms=(
            "DHF",
            "design history file",
            "design controls",
            "21 CFR 820.30",
            "design input",
            "design output",
        ),
        patterns=_compiled(
            r"\bdhf\b",
            r"\bdesign history file\b",
            r"\bdesign controls?\b",
            r"\b21\s*cfr\s*820\.30\b",
        ),
    ),
    AliasGroup(
        name="ecr_dco",
        terms=(
            "ECR",
            "engineering change request",
            "engineering change record",
            "engineering change order",
            "DCO",
            "document change order",
            "design change order",
            "change control",
        ),
        patterns=_compiled(
            r"\becr\b",
            r"\bdco\b",
            r"\bengineering change (?:request|record|order)\b",
            r"\b(?:document|design) change order\b",
            r"\bchange control\b",
        ),
    ),
    AliasGroup(
        name="rmf_rsk",
        terms=(
            "RSK",
            "RMF",
            "risk management file",
            "risk file",
            "risk analysis",
            "hazard analysis",
            "risk management report",
        ),
        patterns=_compiled(
            r"\brsk\b",
            r"\brmf\b",
            r"\brisk management file\b",
            r"\brisk file\b",
            r"\brisk analysis\b",
            r"\bhazard analysis\b",
        ),
    ),
    AliasGroup(
        name="fmea",
        terms=(
            "FMEA",
            "PFMEA",
            "process FMEA",
            "DFMEA",
            "design FMEA",
            "failure modes and effects analysis",
        ),
        patterns=_compiled(
            r"\bfmea\b",
            r"\bpfmea\b",
            r"\bdfmea\b",
            r"\bprocess fmea\b",
            r"\bdesign fmea\b",
            r"\bfailure modes? and effects analysis\b",
        ),
    ),
    AliasGroup(
        name="vvam_traceability",
        terms=(
            "VVAM",
            "verification validation acceptance matrix",
            "V&V matrix",
            "traceability matrix",
            "trace matrix",
            "requirements traceability matrix",
            "requirements trace matrix",
            "TRA",
        ),
        patterns=_compiled(
            r"\bvvam\b",
            r"\bv\s*&\s*v matrix\b",
            r"\bverification validation acceptance matrix\b",
            r"\btraceability matrix\b",
            r"\btrace matrix\b",
            r"\brequirements trace(?:ability)? matrix\b",
            r"\btra\b",
        ),
    ),
    AliasGroup(
        name="third_party",
        terms=(
            "3P",
            "third-party",
            "third party",
            "external laboratory",
            "test lab",
            "certification report",
            "CB report",
        ),
        patterns=_compiled(
            r"\b3\s*p\b",
            r"\bthird[- ]party\b",
            r"\bexternal laborator(?:y|ies)\b",
            r"\btest lab(?:oratory)?\b",
            r"\bcertification report\b",
            r"\bcb report\b",
        ),
    ),
    AliasGroup(
        name="iec_60601",
        terms=(
            "IEC 60601",
            "IEC 60601-1",
            "IEC 60601-1-2",
            "medical electrical equipment",
            "basic safety",
            "essential performance",
            "EMC",
            "electromagnetic compatibility",
        ),
        patterns=_compiled(
            r"\biec\s*60601(?:\s*-\s*1(?:\s*-\s*2)?)?\b",
            r"\b60601(?:\s*-\s*1(?:\s*-\s*2)?)?\b",
            r"\bmedical electrical equipment\b",
            r"\bbasic safety\b",
            r"\bessential performance\b",
            r"\bemc\b",
            r"\belectromagnetic compatibility\b",
        ),
    ),
    AliasGroup(
        name="electrical_safety",
        terms=(
            "electrical safety",
            "leakage current",
            "patient leakage",
            "earth leakage",
            "touch current",
            "dielectric strength",
            "dielectric withstand",
            "hipot",
            "withstand voltage",
            "insulation resistance",
        ),
        patterns=_compiled(
            r"\belectrical safety\b",
            r"\bleakage(?: current)?\b",
            r"\bpatient leakage\b",
            r"\bearth leakage\b",
            r"\btouch current\b",
            r"\bdielectric(?: strength| withstand)?\b",
            r"\bhipot\b",
            r"\bwithstand voltage\b",
            r"\binsulation resistance\b",
        ),
    ),
    AliasGroup(
        name="software_architecture",
        terms=(
            "software architecture",
            "system architecture",
            "architecture diagram",
            "SAD",
            "software design",
            "system design",
            "SOUP",
            "software item",
            "software unit",
            "interface control",
            "architecture description",
        ),
        patterns=_compiled(
            r"\bsoftware architecture\b",
            r"\bsystem architecture\b",
            r"\bsw architecture\b",
            r"\barchitecture diagram\b",
            r"\bsad\b",
            r"\bsoftware design\b",
            r"\bsystem design\b",
            r"\bsoup\b",
            r"\bsoftware item\b",
            r"\bsoftware unit\b",
            r"\binterface control\b",
            r"\barchitecture description\b",
        ),
    ),
)

_ABBREVIATION_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b510(?:\s*\(\s*k\s*\)|\s*k\b)", re.IGNORECASE), "510(k)"),
    (re.compile(r"\bdhf\b", re.IGNORECASE), "DHF"),
    (re.compile(r"\becr\b", re.IGNORECASE), "ECR"),
    (re.compile(r"\bdco\b", re.IGNORECASE), "DCO"),
    (re.compile(r"\brmf\b", re.IGNORECASE), "RMF"),
    (re.compile(r"\brsk\b", re.IGNORECASE), "RSK"),
    (re.compile(r"\bpfmea\b", re.IGNORECASE), "PFMEA"),
    (re.compile(r"\bdfmea\b", re.IGNORECASE), "DFMEA"),
    (re.compile(r"\bfmea\b", re.IGNORECASE), "FMEA"),
    (re.compile(r"\bvvam\b", re.IGNORECASE), "VVAM"),
    (re.compile(r"\b3\s*p\b", re.IGNORECASE), "3P"),
    (re.compile(r"\bemc\b", re.IGNORECASE), "EMC"),
    (re.compile(r"\bsoup\b", re.IGNORECASE), "SOUP"),
    (re.compile(r"\bsad\b", re.IGNORECASE), "SAD"),
)

_IEC_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\biec\s*60601\s*-\s*1\s*-\s*2\b", re.IGNORECASE), "IEC 60601-1-2"),
    (re.compile(r"\biec\s*60601\s*-\s*1\b", re.IGNORECASE), "IEC 60601-1"),
    (re.compile(r"\biec\s*60601\b", re.IGNORECASE), "IEC 60601"),
)


def normalize_query(query: str) -> str:
    normalized = " ".join(query.strip().split())
    normalized = re.sub(
        r"\bk\s*[- ]?(\d{6})\b",
        lambda match: f"K{match.group(1)}",
        normalized,
        flags=re.IGNORECASE,
    )
    for pattern, replacement in _IEC_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    for pattern, replacement in _ABBREVIATION_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    normalized = re.sub(r"\bthird[- ]party\b", "third-party", normalized, flags=re.IGNORECASE)
    return " ".join(normalized.split())


def expand_query(query: str) -> QueryExpansion:
    normalized_query = normalize_query(query)
    expansion_terms: list[str] = []
    matched_groups: list[str] = []
    seen_terms: set[str] = set()

    for group in ALIAS_GROUPS:
        if not _matches_group(normalized_query, group):
            continue
        matched_groups.append(group.name)
        for term in group.terms:
            term_key = term.casefold()
            if term_key in seen_terms or _contains_term(normalized_query, term):
                continue
            seen_terms.add(term_key)
            expansion_terms.append(term)

    expanded_parts = [normalized_query, *expansion_terms] if normalized_query else expansion_terms
    return QueryExpansion(
        original_query=query,
        normalized_query=normalized_query,
        expanded_query=" ".join(expanded_parts),
        terms=tuple(expansion_terms),
        matched_groups=tuple(matched_groups),
    )


def _matches_group(query: str, group: AliasGroup) -> bool:
    return any(pattern.search(query) for pattern in group.patterns)


def _contains_term(query: str, term: str) -> bool:
    normalized_query = _searchable(query)
    normalized_term = _searchable(term)
    return bool(re.search(rf"(?<!\w){re.escape(normalized_term)}(?!\w)", normalized_query))


def _searchable(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


__all__ = [
    "ALIAS_GROUPS",
    "AliasGroup",
    "QueryExpansion",
    "expand_query",
    "normalize_query",
]
