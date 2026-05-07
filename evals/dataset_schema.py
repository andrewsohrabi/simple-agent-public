"""Schema helpers for QMS search evaluation datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


EVALS_DIR = Path(__file__).resolve().parent
CORE_DATASET = EVALS_DIR / "datasets" / "qms_core.jsonl"
SMOKE_DATASET = EVALS_DIR / "datasets" / "qms_smoke.jsonl"

QMS_CATEGORIES = (
    "known_item_retrieval",
    "exploratory_search",
    "compliance_cross_reference",
    "content_extraction_synthesis",
    "revision_change_tracking",
    "cross_document_analysis",
    "enumeration_counting",
)

DIFFICULTIES = ("basic", "intermediate", "advanced")
REQUIRED_RECORD_KEYS = {
    "id",
    "category",
    "prompt",
    "expected",
    "tags",
    "difficulty",
    "dimensions",
}
REQUIRED_EXPECTED_KEYS = {"must_include", "source_ids", "answer_type"}
REQUIRED_DIMENSION_KEYS = {
    "persona",
    "revision_scope",
    "answerability",
    "noise",
    "citation_burden",
}


class DatasetValidationError(ValueError):
    """Raised when an eval dataset does not match the expected schema."""


@dataclass(frozen=True)
class ExpectedEvidence:
    must_include: tuple[str, ...]
    source_ids: tuple[str, ...]
    answer_type: str


@dataclass(frozen=True)
class QmsEvalCase:
    id: str
    category: str
    prompt: str
    expected: ExpectedEvidence
    tags: tuple[str, ...]
    difficulty: str
    dimensions: dict[str, str]

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "QmsEvalCase":
        expected = record["expected"]
        return cls(
            id=record["id"],
            category=record["category"],
            prompt=record["prompt"],
            expected=ExpectedEvidence(
                must_include=tuple(expected["must_include"]),
                source_ids=tuple(expected["source_ids"]),
                answer_type=expected["answer_type"],
            ),
            tags=tuple(record["tags"]),
            difficulty=record["difficulty"],
            dimensions=dict(record["dimensions"]),
        )


def dataset_path(name: str) -> Path:
    """Return a built-in dataset path by short name or JSONL path."""

    if name == "core":
        return CORE_DATASET
    if name == "smoke":
        return SMOKE_DATASET
    return Path(name)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file as dictionaries, rejecting blank or invalid lines."""

    resolved = Path(path)
    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise DatasetValidationError(
                    f"{resolved}:{line_number}: blank lines are not allowed"
                )
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"{resolved}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise DatasetValidationError(
                    f"{resolved}:{line_number}: each line must be a JSON object"
                )
            records.append(record)
    return records


def load_dataset(path: str | Path) -> list[QmsEvalCase]:
    """Load and validate a QMS eval dataset."""

    records = load_jsonl(path)
    return validate_dataset(records)


def validate_dataset(
    records: list[dict[str, Any]],
    *,
    expected_count: int | None = None,
    expected_categories: tuple[str, ...] = QMS_CATEGORIES,
    prompts_per_category: int | None = None,
) -> list[QmsEvalCase]:
    """Validate dataset shape and return typed cases."""

    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    categories: Counter[str] = Counter()

    if expected_count is not None and len(records) != expected_count:
        errors.append(f"expected {expected_count} records, found {len(records)}")

    for index, record in enumerate(records, start=1):
        errors.extend(validate_record(record, index=index))
        record_id = record.get("id")
        prompt = record.get("prompt")
        category = record.get("category")
        if isinstance(record_id, str):
            if record_id in seen_ids:
                errors.append(f"record {index}: duplicate id {record_id!r}")
            seen_ids.add(record_id)
        if isinstance(prompt, str):
            normalized_prompt = " ".join(prompt.lower().split())
            if normalized_prompt in seen_prompts:
                errors.append(f"record {index}: duplicate prompt text")
            seen_prompts.add(normalized_prompt)
        if isinstance(category, str):
            categories[category] += 1

    missing_categories = set(expected_categories) - set(categories)
    extra_categories = set(categories) - set(expected_categories)
    if missing_categories:
        errors.append(f"missing categories: {sorted(missing_categories)}")
    if extra_categories:
        errors.append(f"unknown categories: {sorted(extra_categories)}")

    if prompts_per_category is not None:
        for category in expected_categories:
            count = categories.get(category, 0)
            if count != prompts_per_category:
                errors.append(
                    f"category {category!r} expected {prompts_per_category} records, found {count}"
                )

    if errors:
        raise DatasetValidationError("; ".join(errors))

    return [QmsEvalCase.from_record(record) for record in records]


def validate_record(record: dict[str, Any], *, index: int | None = None) -> list[str]:
    """Return schema errors for one raw record without raising."""

    prefix = f"record {index}: " if index else ""
    errors: list[str] = []

    missing = REQUIRED_RECORD_KEYS - set(record)
    if missing:
        errors.append(f"{prefix}missing keys {sorted(missing)}")

    extra = set(record) - REQUIRED_RECORD_KEYS
    if extra:
        errors.append(f"{prefix}unknown keys {sorted(extra)}")

    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        errors.append(f"{prefix}id must be a non-empty string")
    elif not record_id.startswith("qms_"):
        errors.append(f"{prefix}id must start with 'qms_'")

    category = record.get("category")
    if category not in QMS_CATEGORIES:
        errors.append(f"{prefix}category must be one of {list(QMS_CATEGORIES)}")

    prompt = record.get("prompt")
    if not isinstance(prompt, str) or len(prompt.strip()) < 20:
        errors.append(f"{prefix}prompt must be a string with at least 20 characters")

    tags = record.get("tags")
    if not _is_non_empty_string_list(tags):
        errors.append(f"{prefix}tags must be a non-empty list of strings")
    elif "qms" not in tags:
        errors.append(f"{prefix}tags must include 'qms'")

    difficulty = record.get("difficulty")
    if difficulty not in DIFFICULTIES:
        errors.append(f"{prefix}difficulty must be one of {list(DIFFICULTIES)}")

    dimensions = record.get("dimensions")
    if not isinstance(dimensions, dict):
        errors.append(f"{prefix}dimensions must be an object")
    else:
        missing_dimensions = REQUIRED_DIMENSION_KEYS - set(dimensions)
        if missing_dimensions:
            errors.append(f"{prefix}dimensions missing keys {sorted(missing_dimensions)}")
        extra_dimensions = set(dimensions) - REQUIRED_DIMENSION_KEYS
        if extra_dimensions:
            errors.append(f"{prefix}dimensions has unknown keys {sorted(extra_dimensions)}")
        for key, value in dimensions.items():
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}dimensions.{key} must be a non-empty string")

    expected = record.get("expected")
    if not isinstance(expected, dict):
        errors.append(f"{prefix}expected must be an object")
        return errors

    missing_expected = REQUIRED_EXPECTED_KEYS - set(expected)
    if missing_expected:
        errors.append(f"{prefix}expected missing keys {sorted(missing_expected)}")

    extra_expected = set(expected) - REQUIRED_EXPECTED_KEYS
    if extra_expected:
        errors.append(f"{prefix}expected has unknown keys {sorted(extra_expected)}")

    must_include = expected.get("must_include")
    if not _is_non_empty_string_list(must_include) or len(must_include) < 2:
        errors.append(f"{prefix}expected.must_include must contain at least two strings")

    source_ids = expected.get("source_ids")
    if not _is_non_empty_string_list(source_ids):
        errors.append(f"{prefix}expected.source_ids must be a non-empty list of strings")

    answer_type = expected.get("answer_type")
    if not isinstance(answer_type, str) or not answer_type:
        errors.append(f"{prefix}expected.answer_type must be a non-empty string")

    return errors


def summarize_dataset(cases: list[QmsEvalCase]) -> dict[str, Any]:
    """Return deterministic summary metadata for reporting or tests."""

    category_counts = Counter(case.category for case in cases)
    difficulty_counts = Counter(case.difficulty for case in cases)
    return {
        "total": len(cases),
        "categories": {category: category_counts.get(category, 0) for category in QMS_CATEGORIES},
        "difficulties": {
            difficulty: difficulty_counts.get(difficulty, 0) for difficulty in DIFFICULTIES
        },
    }


def _is_non_empty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )
