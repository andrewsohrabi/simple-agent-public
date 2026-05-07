"""Deterministic scoring and reporting helpers for QMS search evals."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from evals.dataset_schema import QmsEvalCase


TERM_WEIGHT = 0.7
SOURCE_WEIGHT = 0.3
PASS_THRESHOLD = 0.8
COUNT_TERM_KEYWORDS = {
    "active",
    "completed",
    "current",
    "latest",
    "planned",
    "revision",
    "revisions",
    "total",
}
COUNT_TERM_STOPWORDS = {"count"}


@dataclass(frozen=True)
class TableEvidenceRequirement:
    doc_id: str | None
    table_index: int | None
    row_index: int | None


@dataclass(frozen=True)
class ScoreBreakdown:
    id: str
    category: str
    total_score: float
    term_score: float
    source_score: float
    passed: bool
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    matched_sources: tuple[str, ...]
    missing_sources: tuple[str, ...]
    forbidden_terms_present: tuple[str, ...] = ()
    required_backend: str | None = None
    observed_backend: str | None = None
    backend_matched: bool | None = None
    matched_required_doc_ids: tuple[str, ...] = ()
    missing_required_doc_ids: tuple[str, ...] = ()
    matched_table_evidence: tuple[str, ...] = ()
    missing_table_evidence: tuple[str, ...] = ()
    contract_failures: tuple[str, ...] = ()
    top_k_hit: bool | None = None
    recall_at_k: float | None = None
    count_correct: bool | None = None
    citation_validity: float | None = None
    latest_revision_correct: bool | None = None
    obsolete_leakage: bool | None = None
    invalid_citations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(value: str) -> str:
    """Normalize answer text for deterministic substring checks."""

    return re.sub(r"\s+", " ", value.casefold()).strip()


def score_case(
    case: QmsEvalCase,
    answer: str,
    *,
    source_ids: list[str] | tuple[str, ...] | None = None,
    retrieved_source_ids: list[str] | tuple[str, ...] | None = None,
    citations: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    retrieval_backend: str | None = None,
    expected_count: int | None = None,
    reported_count: int | None = None,
    latest_revision: str | None = None,
    retrieved_documents: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    k: int = 5,
    pass_threshold: float = PASS_THRESHOLD,
) -> ScoreBreakdown:
    """Score one answer using exact, deterministic evidence checks."""

    normalized_answer = normalize_text(answer)
    matched_terms = tuple(
        term
        for term in case.expected.must_include
        if _required_term_present(term, normalized_answer)
    )
    missing_terms = tuple(
        term for term in case.expected.must_include if term not in matched_terms
    )
    term_score = _ratio(len(matched_terms), len(case.expected.must_include))
    forbidden_terms_present = tuple(
        term
        for term in case.expected.must_not_include
        if _forbidden_term_present(term, normalized_answer)
    )

    provided_sources = tuple(source_ids or ())
    ranked_sources = tuple(retrieved_source_ids or provided_sources)
    evidence_sources = _evidence_source_ids(
        provided_sources=provided_sources,
        ranked_sources=ranked_sources,
        citations=citations,
        retrieved_documents=retrieved_documents,
    )
    matched_sources = tuple(
        expected
        for expected in case.expected.source_ids
        if _source_present(expected, provided_sources)
    )
    missing_sources = tuple(
        source for source in case.expected.source_ids if source not in matched_sources
    )
    source_score = _ratio(len(matched_sources), len(case.expected.source_ids))
    top_k_sources = ranked_sources[:k]
    top_k_hit = (
        any(_source_present(expected, top_k_sources) for expected in case.expected.source_ids)
        if ranked_sources
        else None
    )
    recall_at_k = (
        _ratio(
            sum(
                1
                for expected in case.expected.source_ids
                if _source_present(expected, top_k_sources)
            ),
            len(case.expected.source_ids),
        )
        if ranked_sources
        else None
    )

    required_count = expected_count
    if required_count is None and case.category == "enumeration_counting":
        required_count = _expected_count_from_terms(case)
    count_correct = None
    if case.category == "enumeration_counting" and required_count is not None:
        observed_count = reported_count if reported_count is not None else _extract_count(answer)
        count_correct = observed_count == required_count

    citation_validity, invalid_citations = _score_citation_validity(
        citations, case.expected.source_ids
    )
    latest_revision_correct = _score_latest_revision(
        case,
        answer,
        latest_revision=latest_revision,
    )
    obsolete_leakage = _score_obsolete_leakage(
        case,
        answer,
        source_ids=provided_sources,
        retrieved_documents=retrieved_documents,
    )
    required_backend = case.expected.required_backend
    observed_backend = (
        retrieval_backend.strip() if isinstance(retrieval_backend, str) else None
    )
    backend_matched = (
        _backend_matches(required_backend, observed_backend)
        if required_backend
        else None
    )
    matched_required_doc_ids = tuple(
        required
        for required in case.expected.required_doc_ids
        if _source_present(required, evidence_sources)
    )
    missing_required_doc_ids = tuple(
        required
        for required in case.expected.required_doc_ids
        if required not in matched_required_doc_ids
    )
    matched_table_evidence = tuple(
        required
        for required in case.expected.required_table_evidence
        if _table_evidence_present(
            required,
            normalized_answer=normalized_answer,
            citations=citations,
            retrieved_documents=retrieved_documents,
            evidence_sources=evidence_sources,
        )
    )
    missing_table_evidence = tuple(
        required
        for required in case.expected.required_table_evidence
        if required not in matched_table_evidence
    )
    contract_failures = _contract_failures(
        forbidden_terms_present=forbidden_terms_present,
        backend_matched=backend_matched,
        missing_required_doc_ids=missing_required_doc_ids,
        missing_table_evidence=missing_table_evidence,
        count_correct=count_correct,
    )

    total_score = round((TERM_WEIGHT * term_score) + (SOURCE_WEIGHT * source_score), 4)
    passed = total_score >= pass_threshold and not contract_failures
    return ScoreBreakdown(
        id=case.id,
        category=case.category,
        total_score=total_score,
        term_score=term_score,
        source_score=source_score,
        passed=passed,
        matched_terms=matched_terms,
        missing_terms=missing_terms,
        matched_sources=matched_sources,
        missing_sources=missing_sources,
        forbidden_terms_present=forbidden_terms_present,
        required_backend=required_backend,
        observed_backend=observed_backend,
        backend_matched=backend_matched,
        matched_required_doc_ids=matched_required_doc_ids,
        missing_required_doc_ids=missing_required_doc_ids,
        matched_table_evidence=matched_table_evidence,
        missing_table_evidence=missing_table_evidence,
        contract_failures=contract_failures,
        top_k_hit=top_k_hit,
        recall_at_k=recall_at_k,
        count_correct=count_correct,
        citation_validity=citation_validity,
        latest_revision_correct=latest_revision_correct,
        obsolete_leakage=obsolete_leakage,
        invalid_citations=invalid_citations,
    )


def aggregate_scores(scores: list[ScoreBreakdown]) -> dict[str, Any]:
    """Aggregate score rows into stable overall and per-category summaries."""

    by_category: dict[str, list[ScoreBreakdown]] = defaultdict(list)
    for score in scores:
        by_category[score.category].append(score)

    return {
        "total": len(scores),
        "passed": sum(1 for score in scores if score.passed),
        "average_score": _average(score.total_score for score in scores),
        "average_term_score": _average(score.term_score for score in scores),
        "average_source_score": _average(score.source_score for score in scores),
        "top_k_hit_rate": _average_bool(score.top_k_hit for score in scores),
        "average_recall_at_k": _average_optional(score.recall_at_k for score in scores),
        "count_accuracy": _average_bool(score.count_correct for score in scores),
        "average_citation_validity": _average_optional(
            score.citation_validity for score in scores
        ),
        "contract_failure_rate": _average_bool(
            bool(score.contract_failures) for score in scores
        ),
        "latest_revision_accuracy": _average_bool(
            score.latest_revision_correct for score in scores
        ),
        "obsolete_leakage_rate": _average_bool(score.obsolete_leakage for score in scores),
        "by_category": {
            category: {
                "total": len(category_scores),
                "passed": sum(1 for score in category_scores if score.passed),
                "pass_rate": _ratio(
                    sum(1 for score in category_scores if score.passed),
                    len(category_scores),
                ),
                "average_score": _average(
                    score.total_score for score in category_scores
                ),
                "top_k_hit_rate": _average_bool(
                    score.top_k_hit for score in category_scores
                ),
                "average_recall_at_k": _average_optional(
                    score.recall_at_k for score in category_scores
                ),
            }
            for category, category_scores in sorted(by_category.items())
        },
    }


def render_markdown_report(
    scores: list[ScoreBreakdown],
    *,
    dataset_name: str,
    pass_threshold: float = PASS_THRESHOLD,
    run_context: dict[str, Any] | None = None,
) -> str:
    """Render a concise Markdown report for humans and PR artifacts."""

    aggregate = aggregate_scores(scores)
    run_context = run_context or {}
    lines = [
        f"# QMS Search Eval Report: {dataset_name}",
        "",
        f"- Timestamp: {run_context.get('timestamp', 'unknown')}",
        f"- Git commit: {run_context.get('git_commit', 'unknown')}",
        f"- Retrieval mode: {run_context.get('retrieval_mode', 'unknown')}",
        f"- Corpus SHA-256: {run_context.get('corpus_hash', 'unknown')}",
        f"- Index manifest: {run_context.get('index_manifest', 'unknown')}",
        f"- Index manifest SHA-256: {run_context.get('index_manifest_hash', 'unknown')}",
        f"- Embedding model: {run_context.get('embedding_model', 'unknown')}",
        f"- Embedding dimensions: {run_context.get('embedding_dimensions', 'unknown')}",
        f"- Vector index: {run_context.get('vector_index', 'unknown')}",
        f"- FAISS type: {run_context.get('faiss_index_type', 'unknown')}",
        f"- Chat model: {run_context.get('chat_model', 'unknown')}",
        f"- Reranker: {run_context.get('reranker_model', 'unknown')}",
        f"- Model config: {_format_model_config(run_context)}",
        f"- Cases: {aggregate['total']}",
        f"- Passed: {aggregate['passed']}",
        f"- Average score: {aggregate['average_score']:.4f}",
        f"- Pass threshold: {pass_threshold:.2f}",
        "",
        "## Retrieval And Answer Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Top-k hit rate | {_format_optional_metric(aggregate['top_k_hit_rate'])} |",
        f"| Recall@k | {_format_optional_metric(aggregate['average_recall_at_k'])} |",
        f"| Count accuracy | {_format_optional_metric(aggregate['count_accuracy'])} |",
        f"| Citation validity | {_format_optional_metric(aggregate['average_citation_validity'])} |",
        f"| Contract failure rate | {_format_optional_metric(aggregate['contract_failure_rate'])} |",
        f"| Latest revision accuracy | {_format_optional_metric(aggregate['latest_revision_accuracy'])} |",
        f"| Obsolete leakage rate | {_format_optional_metric(aggregate['obsolete_leakage_rate'])} |",
        "",
        "## By Category",
        "",
        "| Category | Cases | Passed | Pass Rate | Avg Score | Top-k Hit | Recall@k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, summary in aggregate["by_category"].items():
        lines.append(
            f"| {category} | {summary['total']} | {summary['passed']} | {summary['pass_rate']:.4f} | "
            f"{summary['average_score']:.4f} | {_format_optional_metric(summary['top_k_hit_rate'])} | "
            f"{_format_optional_metric(summary['average_recall_at_k'])} |"
        )

    failures = [score for score in scores if not score.passed]
    if failures:
        lines.extend(["", "## Failing Cases", ""])
        for score in failures:
            lines.append(
                f"- `{score.id}` ({score.category}) score={score.total_score:.4f}; "
                f"missing_terms={list(score.missing_terms)}; "
                f"missing_sources={list(score.missing_sources)}; "
                f"forbidden_terms={list(score.forbidden_terms_present)}; "
                f"contract_failures={list(score.contract_failures)}; "
                f"backend={score.observed_backend}; "
                f"missing_required_doc_ids={list(score.missing_required_doc_ids)}; "
                f"missing_table_evidence={list(score.missing_table_evidence)}; "
                f"top_k_hit={score.top_k_hit}; recall_at_k={score.recall_at_k}; "
                f"count_correct={score.count_correct}; "
                f"latest_revision_correct={score.latest_revision_correct}; "
                f"obsolete_leakage={score.obsolete_leakage}; "
                f"invalid_citations={list(score.invalid_citations)}"
            )

    if failures:
        planned_fixes = planned_fix_classes(failures)
        lines.extend(["", "## Planned Fix Classes", ""])
        for category, count in planned_fixes:
            lines.append(f"- `{category}`: {count} failed cases")
        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- Treat this as a regression/debugging artifact, not a claim that search quality is finished.",
                "- Prioritize deterministic fixes before prompt-only changes: metadata parsing, SQL inventory routing, table row citations, and multi-hop reference following.",
                "- Record repeated failures in `docs/bugs/bugs.md` or `docs/bugs/known_failures.md` before retrying the same fix path a third time.",
            ]
        )

    return "\n".join(lines) + "\n"


def write_json_report(
    path: str | Path,
    scores: list[ScoreBreakdown],
    *,
    run_context: dict[str, Any] | None = None,
) -> None:
    """Write machine-readable score output."""

    payload = {
        "run_context": run_context or {},
        "summary": aggregate_scores(scores),
        "results": [score.to_dict() for score in scores],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_markdown_report(
    path: str | Path,
    scores: list[ScoreBreakdown],
    *,
    dataset_name: str,
    pass_threshold: float = PASS_THRESHOLD,
    run_context: dict[str, Any] | None = None,
) -> None:
    Path(path).write_text(
        render_markdown_report(
            scores,
            dataset_name=dataset_name,
            pass_threshold=pass_threshold,
            run_context=run_context,
        )
    )


def planned_fix_classes(
    failures: list[ScoreBreakdown],
) -> list[tuple[str, int]]:
    """Group failing evals by the likely implementation surface to fix first."""

    counts: dict[str, int] = defaultdict(int)
    for failure in failures:
        if failure.category == "known_item_retrieval":
            counts["known-item exact/metadata routing"] += 1
        elif failure.category == "enumeration_counting":
            counts["SQL-backed enumeration formatting"] += 1
        elif failure.category == "revision_change_tracking":
            counts["revision chain and obsolete/signed handling"] += 1
        elif failure.category == "compliance_cross_reference":
            counts["multi-hop compliance and reference expansion"] += 1
        elif failure.category == "cross_document_analysis":
            counts["cross-document trace expansion"] += 1
        elif failure.category == "content_extraction_synthesis":
            counts["table/section extraction and citation precision"] += 1
        else:
            counts["exploratory recall and grouping"] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 4)


def _required_term_present(term: str, normalized_answer: str) -> bool:
    normalized_term = normalize_text(term)
    if normalized_term in normalized_answer:
        return True
    if not _is_count_like_required_term(normalized_term):
        return False

    numbers = re.findall(r"\b\d+\b", normalized_term)
    if not numbers:
        return False
    if not all(_number_present(number, normalized_answer) for number in numbers):
        return False

    required_words = [
        word
        for word in re.findall(r"[a-z]+", normalized_term)
        if len(word) > 2 and word not in COUNT_TERM_STOPWORDS
    ]
    return all(word in normalized_answer for word in required_words)


def _is_count_like_required_term(normalized_term: str) -> bool:
    if not re.search(r"\b\d+\b", normalized_term):
        return False
    return normalized_term.startswith("count:") or any(
        re.search(rf"\b{re.escape(keyword)}\b", normalized_term)
        for keyword in COUNT_TERM_KEYWORDS
    )


def _number_present(number: str, normalized_answer: str) -> bool:
    return (
        re.search(rf"(?<![\w-]){re.escape(number)}(?![\w-])", normalized_answer)
        is not None
    )


def _expected_count_from_terms(case: QmsEvalCase) -> int | None:
    counts = {
        int(match.group(1))
        for term in case.expected.must_include
        for match in [re.search(r"\bcount\s*:\s*(\d+)\b", term, re.IGNORECASE)]
        if match
    }
    if len(counts) != 1:
        return None
    return next(iter(counts))


def _backend_matches(required: str, observed: str | None) -> bool:
    if not observed:
        return False
    required_normalized = normalize_text(required)
    observed_normalized = normalize_text(observed)
    return required_normalized == observed_normalized or (
        required_normalized in observed_normalized.split(":")
    )


def _contract_failures(
    *,
    forbidden_terms_present: tuple[str, ...],
    backend_matched: bool | None,
    missing_required_doc_ids: tuple[str, ...],
    missing_table_evidence: tuple[str, ...],
    count_correct: bool | None,
) -> tuple[str, ...]:
    failures: list[str] = []
    if forbidden_terms_present:
        failures.append("forbidden_terms_present")
    if backend_matched is False:
        failures.append("required_backend_missing")
    if missing_required_doc_ids:
        failures.append("required_doc_ids_missing")
    if missing_table_evidence:
        failures.append("required_table_evidence_missing")
    if count_correct is False:
        failures.append("required_count_mismatch")
    return tuple(failures)


def _source_present(expected: str, provided_sources: tuple[str, ...]) -> bool:
    expected_normalized = normalize_text(expected)
    for source in provided_sources:
        provided_normalized = normalize_text(source)
        if expected_normalized in provided_normalized:
            return True
    return False


def _score_citation_validity(
    citations: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    expected_sources: tuple[str, ...],
) -> tuple[float | None, tuple[str, ...]]:
    if citations is None:
        return None, ()
    if not citations:
        return 0.0, ()

    invalid: list[str] = []
    valid_count = 0
    for index, citation in enumerate(citations, start=1):
        source = _citation_source_id(citation)
        if not source:
            invalid.append(f"citation_{index}:missing_source")
            continue
        if _source_present(source, expected_sources) or _source_present_any(
            expected_sources, (source,)
        ):
            valid_count += 1
        else:
            invalid.append(source)
    return _ratio(valid_count, len(citations)), tuple(invalid)


def _citation_source_id(citation: dict[str, Any]) -> str:
    for key in ("doc_id", "source_id", "id"):
        value = citation.get(key)
        if isinstance(value, str) and value.strip():
            revision = citation.get("revision")
            if isinstance(revision, str) and revision.strip():
                return f"{value} Rev {revision}"
            return value
    return ""


def _evidence_source_ids(
    *,
    provided_sources: tuple[str, ...],
    ranked_sources: tuple[str, ...],
    citations: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    retrieved_documents: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> tuple[str, ...]:
    sources: list[str] = [*provided_sources, *ranked_sources]
    for citation in citations or ():
        if isinstance(citation, dict):
            sources.append(_citation_source_id(citation))
    for document in retrieved_documents or ():
        if isinstance(document, dict):
            sources.append(_document_source_id(document))
    return _dedupe_non_empty(sources)


def _document_source_id(document: dict[str, Any]) -> str:
    doc_id = document.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        return ""
    revision = document.get("revision")
    if isinstance(revision, str) and revision.strip():
        return f"{doc_id} Rev {revision}"
    return doc_id


def _dedupe_non_empty(values: list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = normalize_text(value)
        if normalized in seen:
            continue
        deduped.append(value)
        seen.add(normalized)
    return tuple(deduped)


def _source_present_any(
    expected_sources: tuple[str, ...],
    provided: tuple[str, ...],
) -> bool:
    return any(_source_present(expected, provided) for expected in expected_sources)


def _table_evidence_present(
    required: str,
    *,
    normalized_answer: str,
    citations: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    retrieved_documents: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    evidence_sources: tuple[str, ...],
) -> bool:
    parsed = _parse_table_evidence_requirement(required)
    if (
        normalize_text(required) in normalized_answer
        and _table_requirement_source_present(parsed, evidence_sources)
    ):
        return True

    records = [
        record
        for record in [*(citations or ()), *(retrieved_documents or ())]
        if isinstance(record, dict)
    ]
    if parsed and any(
        _record_matches_table_requirement(record, parsed) for record in records
    ):
        return True

    labels = " ".join(_table_evidence_label(record) for record in records)
    return normalize_text(required) in normalize_text(labels)


def _parse_table_evidence_requirement(required: str) -> TableEvidenceRequirement | None:
    match = re.search(
        r"(?:(?P<doc>[A-Z0-9]+(?:-[A-Z0-9]+)+)\s+)?"
        r"Table\s+(?P<table>\d+)"
        r"(?:\s+row\s+(?P<row>\d+))?",
        required,
        re.IGNORECASE,
    )
    if not match:
        return None
    row = match.group("row")
    return TableEvidenceRequirement(
        doc_id=match.group("doc").upper() if match.group("doc") else None,
        table_index=int(match.group("table")),
        row_index=int(row) if row else None,
    )


def _table_requirement_source_present(
    parsed: TableEvidenceRequirement | None,
    evidence_sources: tuple[str, ...],
) -> bool:
    if parsed is None or parsed.doc_id is None:
        return True
    return _source_present(parsed.doc_id, evidence_sources)


def _record_matches_table_requirement(
    record: dict[str, Any],
    required: TableEvidenceRequirement,
) -> bool:
    if required.doc_id and not _source_present(
        required.doc_id, (_record_source_id(record),)
    ):
        return False

    table_index = _record_table_index(record)
    if required.table_index is not None and table_index != required.table_index:
        return False

    if required.row_index is None:
        return True
    row_start, row_end = _record_row_range(record)
    if row_start is not None and row_end is not None:
        return row_start <= required.row_index <= row_end
    label = normalize_text(_table_evidence_label(record))
    return re.search(rf"\brow\s+{required.row_index}\b", label) is not None


def _record_source_id(record: dict[str, Any]) -> str:
    citation_source = _citation_source_id(record)
    if citation_source:
        return citation_source
    return _document_source_id(record)


def _record_table_index(record: dict[str, Any]) -> int | None:
    direct = _optional_int(record.get("table_index"))
    if direct is not None:
        return direct
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        direct = _optional_int(metadata.get("table_index"))
        if direct is not None:
            return direct
    match = re.search(r"\bTable\s+(\d+)\b", _record_raw_label(record), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _record_row_range(record: dict[str, Any]) -> tuple[int | None, int | None]:
    row_start = _optional_int(record.get("row_start"))
    row_end = _optional_int(record.get("row_end"))
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        row_start = (
            row_start
            if row_start is not None
            else _optional_int(metadata.get("row_start"))
        )
        row_end = (
            row_end
            if row_end is not None
            else _optional_int(metadata.get("row_end"))
        )
    if row_start is not None and row_end is None:
        row_end = row_start
    if row_end is not None and row_start is None:
        row_start = row_end
    return row_start, row_end


def _table_evidence_label(record: dict[str, Any]) -> str:
    values = [_record_raw_label(record)]
    row_start, row_end = _record_row_range(record)
    table_index = _record_table_index(record)
    if table_index is not None:
        values.append(f"Table {table_index}")
    if row_start is not None and row_end is not None:
        if row_start == row_end:
            values.append(f"Row {row_start}")
        else:
            values.append(f"Rows {row_start}-{row_end}")
    return " ".join(value for value in values if value)


def _record_raw_label(record: dict[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "doc_id",
        "source_id",
        "id",
        "section",
        "chunk_id",
        "evidence_type",
        "text",
    ):
        value = record.get(key)
        if isinstance(value, str):
            values.append(value)
    heading_path = record.get("heading_path")
    if isinstance(heading_path, (list, tuple)):
        values.extend(str(item) for item in heading_path)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        values.extend(
            str(metadata.get(key) or "")
            for key in ("document_code", "section", "kind", "evidence_type")
        )
        metadata_heading = metadata.get("heading_path")
        if isinstance(metadata_heading, (list, tuple)):
            values.extend(str(item) for item in metadata_heading)
    row_cells = record.get("row_cells")
    if isinstance(row_cells, dict):
        values.extend(str(value) for value in row_cells.values())
    return " ".join(value for value in values if value)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _score_latest_revision(
    case: QmsEvalCase,
    answer: str,
    *,
    latest_revision: str | None,
) -> bool | None:
    if case.dimensions.get("revision_scope") not in {"latest", "all_revisions"}:
        return None
    expected_revision = _expected_revision(case)
    if not expected_revision:
        return None
    observed = latest_revision or _first_revision(answer)
    if not observed:
        return False
    return normalize_text(expected_revision) == normalize_text(observed)


def _expected_revision(case: QmsEvalCase) -> str | None:
    for term in case.expected.must_include:
        revision = _first_revision(term)
        if revision:
            return revision
    return None


def _first_revision(value: str) -> str | None:
    match = re.search(r"\bRev(?:ision)?\s+([A-Z0-9]+)\b", value, re.IGNORECASE)
    if not match:
        return None
    return f"Rev {match.group(1).upper()}"


def _score_obsolete_leakage(
    case: QmsEvalCase,
    answer: str,
    *,
    source_ids: tuple[str, ...],
    retrieved_documents: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> bool | None:
    revision_scope = case.dimensions.get("revision_scope")
    expects_obsolete = revision_scope == "obsolete_explicit" or any(
        "obsolete" in normalize_text(term) for term in case.expected.must_include
    )
    if expects_obsolete:
        return None
    if revision_scope not in {"latest", "all", "all_revisions"}:
        return None
    if "obsolete" in normalize_text(answer):
        return True
    if any("obsolete" in normalize_text(source) for source in source_ids):
        return True
    for document in retrieved_documents or ():
        metadata = document.get("metadata")
        if isinstance(metadata, dict) and metadata.get("is_obsolete") is True:
            return True
        title = document.get("title")
        if isinstance(title, str) and "obsolete" in normalize_text(title):
            return True
    return False


def _forbidden_term_present(term: str, normalized_answer: str) -> bool:
    normalized_term = normalize_text(term)
    if re.fullmatch(r"[a-z0-9]{1,4}", normalized_term):
        return (
            re.search(rf"\b{re.escape(normalized_term)}\b", normalized_answer)
            is not None
        )
    return normalized_term in normalized_answer


def _extract_count(answer: str) -> int | None:
    match = re.search(r"\b(?:count\s*:\s*)?(\d+)\b", answer, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _average(values: Any) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return round(sum(collected) / len(collected), 4)


def _average_optional(values: Any) -> float | None:
    collected = [value for value in values if value is not None]
    if not collected:
        return None
    return round(sum(collected) / len(collected), 4)


def _average_bool(values: Any) -> float | None:
    collected = [value for value in values if value is not None]
    if not collected:
        return None
    return round(sum(1 for value in collected if value) / len(collected), 4)


def _format_optional_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_model_config(run_context: dict[str, Any]) -> str:
    model_config = run_context.get("model_config")
    if isinstance(model_config, dict) and model_config:
        return ", ".join(
            f"{key}={value}" for key, value in sorted(model_config.items())
        )

    fallback = {
        "agent_model": run_context.get("agent_model"),
        "chat_model": run_context.get("chat_model"),
        "embedding_model": run_context.get("embedding_model"),
        "query_model": run_context.get("query_model"),
        "reranker_model": run_context.get("reranker_model"),
    }
    fallback = {
        key: value
        for key, value in fallback.items()
        if value not in (None, "", "unknown")
    }
    if not fallback:
        return "unknown"
    return ", ".join(f"{key}={value}" for key, value in sorted(fallback.items()))
