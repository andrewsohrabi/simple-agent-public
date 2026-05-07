import copy

import pytest

from evals.dataset_schema import (
    CORE_DATASET,
    QMS_CATEGORIES,
    SMOKE_DATASET,
    DatasetValidationError,
    QmsEvalCase,
    load_jsonl,
    validate_dataset,
    validate_record,
)
from evals.reporting import aggregate_scores, score_case


def test_qms_core_dataset_has_expected_shape():
    records = load_jsonl(CORE_DATASET)
    cases = validate_dataset(
        records,
        expected_count=84,
        prompts_per_category=12,
    )

    assert len(cases) == 84
    assert {case.category for case in cases} == set(QMS_CATEGORIES)
    assert len({case.id for case in cases}) == 84
    assert len({case.prompt for case in cases}) == 84


def test_qms_smoke_dataset_has_one_case_per_category_and_uses_core_ids():
    core_ids = {record["id"] for record in load_jsonl(CORE_DATASET)}
    smoke_records = load_jsonl(SMOKE_DATASET)
    smoke_cases = validate_dataset(
        smoke_records,
        expected_count=7,
        prompts_per_category=1,
    )

    assert {case.category for case in smoke_cases} == set(QMS_CATEGORIES)
    assert {case.id for case in smoke_cases}.issubset(core_ids)
    assert all("smoke" in case.tags for case in smoke_cases)


def test_validate_record_rejects_malformed_expected_schema():
    record = copy.deepcopy(load_jsonl(SMOKE_DATASET)[0])
    record["expected"].pop("source_ids")

    errors = validate_record(record, index=1)

    assert any("expected missing keys" in error for error in errors)
    with pytest.raises(DatasetValidationError):
        validate_dataset([record])


def test_scoring_is_deterministic_for_terms_and_sources():
    record = load_jsonl(SMOKE_DATASET)[0]
    assert validate_record(record) == []
    case = QmsEvalCase.from_record(record)
    score = score_case(
        case,
        "The latest MX1 Bill of Materials is BOM-055 Rev G.",
        source_ids=["BOM-055 - MX1 Top-level assembly"],
    )

    assert score.total_score == 1.0
    assert score.passed
    assert aggregate_scores([score])["average_score"] == 1.0
