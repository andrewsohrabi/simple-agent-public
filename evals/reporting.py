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
    pass_threshold: float = PASS_THRESHOLD,
) -> ScoreBreakdown:
    """Score one answer using exact, deterministic evidence checks."""

    normalized_answer = normalize_text(answer)
    matched_terms = tuple(
        term
        for term in case.expected.must_include
        if normalize_text(term) in normalized_answer
    )
    missing_terms = tuple(
        term for term in case.expected.must_include if term not in matched_terms
    )
    term_score = _ratio(len(matched_terms), len(case.expected.must_include))

    provided_sources = tuple(source_ids or ())
    matched_sources = tuple(
        expected
        for expected in case.expected.source_ids
        if _source_present(expected, provided_sources)
    )
    missing_sources = tuple(
        source for source in case.expected.source_ids if source not in matched_sources
    )
    source_score = _ratio(len(matched_sources), len(case.expected.source_ids))

    total_score = round((TERM_WEIGHT * term_score) + (SOURCE_WEIGHT * source_score), 4)
    return ScoreBreakdown(
        id=case.id,
        category=case.category,
        total_score=total_score,
        term_score=term_score,
        source_score=source_score,
        passed=total_score >= pass_threshold,
        matched_terms=matched_terms,
        missing_terms=missing_terms,
        matched_sources=matched_sources,
        missing_sources=missing_sources,
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
        "by_category": {
            category: {
                "total": len(category_scores),
                "passed": sum(1 for score in category_scores if score.passed),
                "average_score": _average(
                    score.total_score for score in category_scores
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
        f"- Embedding model: {run_context.get('embedding_model', 'unknown')}",
        f"- Embedding dimensions: {run_context.get('embedding_dimensions', 'unknown')}",
        f"- Vector index: {run_context.get('vector_index', 'unknown')}",
        f"- FAISS type: {run_context.get('faiss_index_type', 'unknown')}",
        f"- Chat model: {run_context.get('chat_model', 'unknown')}",
        f"- Reranker: {run_context.get('reranker_model', 'unknown')}",
        f"- Cases: {aggregate['total']}",
        f"- Passed: {aggregate['passed']}",
        f"- Average score: {aggregate['average_score']:.4f}",
        f"- Pass threshold: {pass_threshold:.2f}",
        "",
        "## By Category",
        "",
        "| Category | Cases | Passed | Avg Score |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, summary in aggregate["by_category"].items():
        lines.append(
            f"| {category} | {summary['total']} | {summary['passed']} | {summary['average_score']:.4f} |"
        )

    failures = [score for score in scores if not score.passed]
    if failures:
        lines.extend(["", "## Failing Cases", ""])
        for score in failures:
            lines.append(
                f"- `{score.id}` ({score.category}) score={score.total_score:.4f}; "
                f"missing_terms={list(score.missing_terms)}; "
                f"missing_sources={list(score.missing_sources)}"
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


def _source_present(expected: str, provided_sources: tuple[str, ...]) -> bool:
    expected_normalized = normalize_text(expected)
    for source in provided_sources:
        provided_normalized = normalize_text(source)
        if expected_normalized in provided_normalized:
            return True
    return False


def _average(values: Any) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return round(sum(collected) / len(collected), 4)
