from evals.reporting import _extract_count as reporting_extract_count
from evals.run_search_evals import _extract_count as harness_extract_count


def test_count_extractors_prefer_explicit_count_after_document_id():
    answer = "BOM-055 is the relevant family. Count: 3 active current records."

    assert reporting_extract_count(answer) == 3
    assert harness_extract_count(answer) == 3


def test_count_extractors_ignore_document_id_numbers_without_count():
    answer = "BOM-055 is the relevant family, but the answer does not report a count."

    assert reporting_extract_count(answer) is None
    assert harness_extract_count(answer) is None


def test_count_extractors_keep_single_unambiguous_number_fallback():
    answer = "There are 3 signed ECR records in the inventory."

    assert reporting_extract_count(answer) == 3
    assert harness_extract_count(answer) == 3
