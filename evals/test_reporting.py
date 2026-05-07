import hashlib

from evals.dataset_schema import ExpectedEvidence, QmsEvalCase
from evals.reporting import aggregate_scores, render_markdown_report, score_case
from evals.run_search_evals import _sha256_file


def test_reports_include_run_metadata_and_category_pass_summary():
    cases = [
        QmsEvalCase(
            id="qms_known_item_retrieval_001",
            category="known_item_retrieval",
            prompt="Find the latest active BOM record for the MX1 device.",
            expected=ExpectedEvidence(
                must_include=("BOM-055",),
                source_ids=("BOM-055",),
                answer_type="source_grounded_answer",
            ),
            tags=("qms",),
            difficulty="basic",
            dimensions={},
        ),
        QmsEvalCase(
            id="qms_known_item_retrieval_002",
            category="known_item_retrieval",
            prompt="Find the MX1 project quality planning record.",
            expected=ExpectedEvidence(
                must_include=("Project Quality Plan",),
                source_ids=("PLN-P01-062",),
                answer_type="source_grounded_answer",
            ),
            tags=("qms",),
            difficulty="basic",
            dimensions={},
        ),
    ]
    scores = [
        score_case(
            cases[0],
            "The matching record is BOM-055.",
            source_ids=["BOM-055 Rev G"],
        ),
        score_case(cases[1], "No matching record found.", source_ids=[]),
    ]

    summary = aggregate_scores(scores)
    markdown = render_markdown_report(
        scores,
        dataset_name="smoke",
        run_context={
            "timestamp": "2026-05-07-020000",
            "git_commit": "abc123def456",
            "retrieval_mode": "local",
            "corpus_hash": "corpus-sha",
            "index_manifest": ".data/qms-index/manifest.json",
            "index_manifest_hash": "manifest-sha",
            "model_config": {
                "chat_model": "gpt-5.5",
                "embedding_model": "text-embedding-3-large",
                "query_model": "gpt-5.4-mini",
            },
        },
    )

    assert summary["by_category"]["known_item_retrieval"] == {
        "total": 2,
        "passed": 1,
        "pass_rate": 0.5,
        "average_score": 0.5,
        "top_k_hit_rate": 1.0,
        "average_recall_at_k": 1.0,
    }
    assert "- Git commit: abc123def456" in markdown
    assert "- Corpus SHA-256: corpus-sha" in markdown
    assert "- Index manifest SHA-256: manifest-sha" in markdown
    assert "- Retrieval mode: local" in markdown
    assert (
        "- Model config: chat_model=gpt-5.5, "
        "embedding_model=text-embedding-3-large, query_model=gpt-5.4-mini"
    ) in markdown
    assert "| known_item_retrieval | 2 | 1 | 0.5000 | 0.5000 |" in markdown


def test_sha256_file_returns_digest_or_unknown(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"chunk_count": 1}\n')

    assert _sha256_file(manifest) == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert _sha256_file(tmp_path / "missing.json") == "unknown"


def test_scores_trace_metrics_without_requiring_live_retrieval():
    case = QmsEvalCase(
        id="qms_revision_change_tracking_003",
        category="revision_change_tracking",
        prompt="What is the latest active revision of BOM-055 and which older revisions exist?",
        expected=ExpectedEvidence(
            must_include=("BOM-055", "Rev G"),
            source_ids=("BOM-055",),
            answer_type="source_grounded_answer",
        ),
        tags=("qms",),
        difficulty="advanced",
        dimensions={
            "persona": "medical_writer_rd",
            "revision_scope": "all_revisions",
            "answerability": "answerable",
            "noise": "partial_id",
            "citation_burden": "table_or_section",
        },
    )

    score = score_case(
        case,
        "BOM-055 Rev G is the latest active revision.",
        source_ids=["BOM-055 Rev G"],
        retrieved_source_ids=["RSK Rev C", "BOM-055 Rev G", "BOM-055 Rev F"],
        citations=[{"doc_id": "BOM-055", "revision": "G"}],
        latest_revision="Rev G",
        retrieved_documents=[
            {
                "doc_id": "BOM-055",
                "revision": "G",
                "title": "Bill of Materials",
                "metadata": {"is_latest": True, "is_obsolete": False},
            }
        ],
    )

    summary = aggregate_scores([score])
    markdown = render_markdown_report([score], dataset_name="trace")

    assert score.top_k_hit is True
    assert score.recall_at_k == 1.0
    assert score.citation_validity == 1.0
    assert score.latest_revision_correct is True
    assert score.obsolete_leakage is False
    assert summary["top_k_hit_rate"] == 1.0
    assert summary["average_recall_at_k"] == 1.0
    assert summary["latest_revision_accuracy"] == 1.0
    assert "| Top-k hit rate | 1.0000 |" in markdown
    assert "| revision_change_tracking | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |" in markdown


def test_count_and_obsolete_metrics_are_deterministic_guardrails():
    count_case = QmsEvalCase(
        id="qms_enumeration_counting_001",
        category="enumeration_counting",
        prompt="How many engineering change requests are in the system?",
        expected=ExpectedEvidence(
            must_include=("ECR", "count"),
            source_ids=("ECR",),
            answer_type="source_grounded_answer",
        ),
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
    latest_case = QmsEvalCase(
        id="qms_known_item_retrieval_001",
        category="known_item_retrieval",
        prompt="Find the Bill of Materials for the MX1 system and return the latest active revision.",
        expected=ExpectedEvidence(
            must_include=("BOM-055", "Rev G"),
            source_ids=("BOM-055",),
            answer_type="source_grounded_answer",
        ),
        tags=("qms",),
        difficulty="basic",
        dimensions={
            "persona": "regulatory_affairs",
            "revision_scope": "latest",
            "answerability": "answerable",
            "noise": "none",
            "citation_burden": "single_source",
        },
    )

    count_score = score_case(
        count_case,
        "Count: 3 ECR document revisions.",
        source_ids=["ECR-593"],
        expected_count=3,
    )
    leakage_score = score_case(
        latest_case,
        "BOM-055 Rev G is current, but obsolete Rev F also appears in context.",
        source_ids=["BOM-055 Rev G", "BOM-055 Rev F obsolete"],
    )
    summary = aggregate_scores([count_score, leakage_score])

    assert count_score.count_correct is True
    assert leakage_score.obsolete_leakage is True
    assert summary["count_accuracy"] == 1.0
    assert summary["obsolete_leakage_rate"] == 0.5
