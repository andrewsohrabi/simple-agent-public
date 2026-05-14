from evals.dataset_schema import ExpectedEvidence, QmsEvalCase
from evals.reporting import score_case


def _case(
    *,
    category: str = "known_item_retrieval",
    expected: ExpectedEvidence,
) -> QmsEvalCase:
    return QmsEvalCase(
        id="qms_contract_enforcement_001",
        category=category,
        prompt="Return a source-grounded QMS answer for this contract check.",
        expected=expected,
        tags=("qms",),
        difficulty="basic",
        dimensions={
            "persona": "regulatory_affairs",
            "revision_scope": "all",
            "answerability": "answerable",
            "noise": "none",
            "citation_burden": "single_source",
        },
    )


def test_required_backend_missing_fails_otherwise_correct_case():
    case = _case(
        category="exploratory_search",
        expected=ExpectedEvidence(
            must_include=("VVPR", "93 total"),
            source_ids=("VVPR",),
            answer_type="source_grounded_answer",
            required_backend="sql_inventory",
        ),
    )

    score = score_case(
        case,
        "The VVPR inventory has 93 total revisions.",
        source_ids=["VVPR-P01-179 Rev B"],
    )

    assert score.total_score == 1.0
    assert score.backend_matched is False
    assert score.contract_failures == ("required_backend_missing",)
    assert not score.passed


def test_required_backend_match_satisfies_contract_gate():
    case = _case(
        category="exploratory_search",
        expected=ExpectedEvidence(
            must_include=("VVPR", "93 total"),
            source_ids=("VVPR",),
            answer_type="source_grounded_answer",
            required_backend="sql_inventory",
        ),
    )

    score = score_case(
        case,
        "The VVPR inventory has 93 total revisions.",
        source_ids=["VVPR-P01-179 Rev B"],
        retrieval_backend="sql_inventory",
    )

    assert score.backend_matched is True
    assert score.contract_failures == ()
    assert score.passed


def test_required_doc_ids_missing_fail_even_when_source_family_matches():
    case = _case(
        expected=ExpectedEvidence(
            must_include=("510(k)", "DHF"),
            source_ids=("MEMO",),
            answer_type="source_grounded_answer",
            required_doc_ids=("MEMO-P01-859", "DHF-008"),
        ),
    )

    score = score_case(
        case,
        "The 510(k) summary is filed with DHF evidence.",
        source_ids=["MEMO-P01-859 Rev A"],
    )

    assert score.total_score == 1.0
    assert score.missing_required_doc_ids == ("DHF-008",)
    assert score.contract_failures == ("required_doc_ids_missing",)
    assert not score.passed


def test_required_table_evidence_missing_fails_otherwise_correct_case():
    case = _case(
        category="content_extraction_synthesis",
        expected=ExpectedEvidence(
            must_include=("IEC 60601-1", "PASS"),
            source_ids=("MEMO-P01-685",),
            answer_type="source_grounded_answer",
            required_table_evidence=("MEMO-P01-685 Table 2 row 2",),
        ),
    )

    score = score_case(
        case,
        "The electrical safety acceptance criterion references IEC 60601-1 and PASS.",
        source_ids=["MEMO-P01-685 Rev A"],
        citations=[{"doc_id": "MEMO-P01-685", "revision": "A"}],
    )

    assert score.total_score == 1.0
    assert score.missing_table_evidence == ("MEMO-P01-685 Table 2 row 2",)
    assert score.contract_failures == ("required_table_evidence_missing",)
    assert not score.passed


def test_required_table_evidence_passes_with_structured_row_citation():
    case = _case(
        category="content_extraction_synthesis",
        expected=ExpectedEvidence(
            must_include=("IEC 60601-1", "PASS"),
            source_ids=("MEMO-P01-685",),
            answer_type="source_grounded_answer",
            required_table_evidence=("MEMO-P01-685 Table 2 row 2",),
        ),
    )

    score = score_case(
        case,
        "The electrical safety acceptance criterion references IEC 60601-1 and PASS.",
        source_ids=["MEMO-P01-685 Rev A"],
        citations=[
            {
                "doc_id": "MEMO-P01-685",
                "revision": "A",
                "table_index": 2,
                "row_start": 2,
                "row_end": 2,
            }
        ],
    )

    assert score.matched_table_evidence == ("MEMO-P01-685 Table 2 row 2",)
    assert score.contract_failures == ()
    assert score.passed


def test_explicit_count_term_is_scored_as_required_count():
    case = _case(
        category="enumeration_counting",
        expected=ExpectedEvidence(
            must_include=("Count: 3", "signed"),
            source_ids=("ECR",),
            answer_type="source_grounded_answer",
        ),
    )

    score = score_case(
        case,
        "There are 3 signed ECR records in the inventory.",
        source_ids=["ECR-577 Rev A"],
    )

    assert score.matched_terms == ("Count: 3", "signed")
    assert score.count_correct is True
    assert score.contract_failures == ()
    assert score.passed
