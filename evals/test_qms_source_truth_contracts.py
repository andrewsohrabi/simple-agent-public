from __future__ import annotations

from pathlib import Path
import re

import pytest
from dotenv import load_dotenv

from agent.config import load_config
from agent.search.service import QmsSearchService


INDEX_DB = Path(".data/qms-index/qms.sqlite")


@pytest.fixture(scope="module")
def qms_service() -> QmsSearchService:
    if not INDEX_DB.exists():
        pytest.skip("local QMS index is not available")
    load_dotenv(".env")
    return QmsSearchService(load_config(), use_hash_embeddings=True)


@pytest.mark.parametrize(
    ("query", "intent", "required_terms", "required_sources", "forbidden_terms"),
    [
        (
            "Find the Bill of Materials for the MX1 system",
            "mx1_bom",
            ("BOM-055", "Rev G"),
            ("BOM-055",),
            (),
        ),
        (
            "Where is the 510(k) summary for the device?",
            "510k_summary_location",
            ("K241567", "standalone indexed document"),
            ("MEMO-P01-859", "DHF-008", "PLN-P01-061"),
            (),
        ),
        (
            "What verification test protocols do we have for the MX1?",
            "vvpr_inventory",
            ("93 total VVPR revisions", "89 latest active"),
            ("VVPR",),
            (),
        ),
        (
            "Show me all risk-related documents",
            "risk_related_inventory",
            (
                "Risk-related inventory",
                "Current active RSK records",
                "PLN-P01-063",
                "Obsolete/historical",
                "RSK-P01-010",
                "RSK-P01-011",
                "RSK-P01-012",
            ),
            ("RSK-P01-017", "PLN-P01-063", "VVAM-P01-004"),
            (),
        ),
        (
            "Does our Design History File include everything required by FDA 21 CFR 820.30?",
            "dhf_82030",
            ("DHF-008", "820.30", "QMSR"),
            ("DHF-008", "PLN-P01-062"),
            (),
        ),
        (
            "Which verification protocols trace back to the risk analysis?",
            "risk_protocol_trace",
            ("Current P01 verification protocols", "VVAM-P01-004", "MEMO-P01-630"),
            ("RSK-P01-017", "VVAM-P01-004", "MEMO-P01-630", "VVPR-P01-179"),
            (
                "VVPR-P00",
                "TRA",
                "VVPR-P01-159",
                "VVPR-P01-162",
                "VVPR-P01-166",
                "VVPR-P01-168",
                "VVPR-P01-177",
                "VVPR-P01-183",
                "VVPR-P01-205",
            ),
        ),
        (
            "What are the acceptance criteria for the electrical safety verification test?",
            "electrical_safety_acceptance",
            ("MEMO-P01-685 Table 2 row 2", "IEC 60601-1", "PASS"),
            ("MEMO-P01-685", "3P-P01-33"),
            (),
        ),
        (
            "Summarize all design review action items that are still open",
            "open_design_review_actions",
            ("MEMO-P01-859", "Section 4", "open design-review action items"),
            ("MEMO-P01-859",),
            ("MEMO-P01-685",),
        ),
        (
            "What changed between Rev C and Rev D of the risk analysis?",
            "ambiguous_risk_revision_diff",
            ("no single RSK document chain", "no indexed RSK Rev D"),
            ("RSK",),
            ("Security Risk Assessment comparison",),
        ),
        (
            "Show me all ECRs filed in the last year and their status",
            "ecr_last_year_status",
            ("current-date policy", "approval effective dates", "status active/current"),
            ("ECR-577", "ECR-587", "ECR-593"),
            (),
        ),
        (
            "Trace the requirement for electrical leakage testing from the risk file through to the verification report",
            "electrical_leakage_trace",
            ("RSK_R230", "3P-P01-33", "PASS"),
            ("RSK-P01-010", "RSK-P01-017", "VVAM-P01-004", "MEMO-P01-685", "3P-P01-33"),
            ("VVPR-P00", "TRA"),
        ),
        (
            "Map all third-party test reports to the regulatory requirements they satisfy",
            "third_party_report_mapping",
            ("3P-P01-32", "3P-P01-33", "IEC 60601-1-2", "IEC 60601-1"),
            ("3P-P01-32", "3P-P01-33", "MEMO-P01-685", "VVAM-P01-004"),
            ("TRA",),
        ),
        (
            "How many engineering change requests are in the system?",
            "ecr_count",
            ("Count: 3", "signed", "not obsolete"),
            ("ECR-577", "ECR-587", "ECR-593"),
            (),
        ),
        (
            "How many verification protocols have we completed vs. planned?",
            "verification_completed_vs_planned",
            ("67 completed", "86 planned", "MEMO-P01-685", "PLN-P01-065"),
            ("MEMO-P01-685", "PLN-P01-065"),
            (),
        ),
    ],
)
def test_audited_queries_return_source_truth(
    qms_service: QmsSearchService,
    query: str,
    intent: str,
    required_terms: tuple[str, ...],
    required_sources: tuple[str, ...],
    forbidden_terms: tuple[str, ...],
) -> None:
    result = qms_service.search(query, mode="local", limit=8)
    answer = str(result["answer"])
    cited_sources = [
        str(citation.get("doc_id", ""))
        for citation in result.get("citations", [])
        if isinstance(citation, dict)
    ]

    assert result["query_plan"]["intent"] == intent
    for term in required_terms:
        assert term.casefold() in answer.casefold()
    for source in required_sources:
        assert any(source.casefold() in cited.casefold() for cited in cited_sources)
    for term in forbidden_terms:
        assert not _forbidden_present(term, answer)


def test_risk_protocol_trace_uses_explicit_row_level_evidence(qms_service: QmsSearchService) -> None:
    result = qms_service.search(
        "Which verification protocols trace back to the risk analysis?",
        mode="local",
        limit=8,
    )
    answer = str(result["answer"])

    for supported_id in (
        "VVPR-P01-160",
        "VVPR-P01-165",
        "VVPR-P01-179",
        "VVPR-P01-182",
        "VVPR-P01-184",
        "VVPR-P01-199",
        "VVPR-P01-200",
        "VVPR-P01-203",
        "VVPR-P01-204",
        "VVPR-P01-219",
        "VVPR-P01-229",
    ):
        assert supported_id in answer

    for unsupported_id in (
        "VVPR-P01-159",
        "VVPR-P01-162",
        "VVPR-P01-166",
        "VVPR-P01-168",
        "VVPR-P01-177",
        "VVPR-P01-183",
        "VVPR-P01-205",
        "VVPR-P00",
        "TRA",
    ):
        assert not _forbidden_present(unsupported_id, answer)

    assert "VVPR-P01-100" in answer
    assert "not indexed" in answer.casefold()


@pytest.mark.parametrize(
    ("query", "intent", "required_terms", "required_sources", "forbidden_terms"),
    [
        (
            "Find the signed MX1 software development configuration management memo.",
            "software_config_management_memo",
            ("MEMO-P01-638", "Software Development Configuration Management", "signed"),
            ("MEMO-P01-638",),
            ("I could not find",),
        ),
        (
            "Find the MX1 System Architecture Diagram memo.",
            "system_architecture_diagram_memo",
            ("MEMO-P01-658", "System Architecture Diagram"),
            ("MEMO-P01-658",),
            (),
        ),
        (
            "Compare Rev B and Rev C records for collimation or beam-angle verification if both are present.",
            "collimation_beam_angle_revision_compare",
            ("Rev B", "Rev C", "VVPR-P01-189", "VVPR-P01-214"),
            ("VVPR-P01-189", "VVPR-P01-214"),
            ("BOM-055", "3P-P01-32", "3P-P01-33"),
        ),
        (
            "Trace software critical fault handling from risk controls to verification evidence.",
            "software_critical_fault_trace",
            ("critical faults", "risk", "VVPR-P01-181"),
            ("RSK", "VVPR-P01-181"),
            (),
        ),
        (
            "Trace software acquisition verification across MX1 software system protocol reports.",
            "software_acquisition_trace",
            ("Radiographic", "Radioscopic", "VVPR-P01-179"),
            ("MEMO-P01-630", "VVPR-P01-179"),
            (),
        ),
        (
            "Trace pediatric filtration from requirements or risk rationale through verification evidence.",
            "pediatric_filtration_trace",
            ("pediatric filtration", "VVPR-P01-152"),
            ("DR-P01-005", "RSK-P01-010", "VVAM-P01-004", "VVPR-P01-152"),
            ("TRA-024", "TRA-025", "TRA-026", "Customer Training"),
        ),
        (
            "How many traceability matrices are in the corpus?",
            "traceability_matrix_count",
            ("1 current", "VVAM-P01-004"),
            ("VVAM-P01-004",),
            ("TRA-024", "TRA-025", "TRA-026", "Customer Training"),
        ),
        (
            "How many non-empty DOCX records were ingested after ignoring empty documents?",
            "ingest_manifest_count",
            ("non-empty", "ingested", "ingest_manifest"),
            ("ingest_manifest",),
            (),
        ),
    ],
)
def test_strict_eval_regressions_are_source_grounded(
    qms_service: QmsSearchService,
    query: str,
    intent: str,
    required_terms: tuple[str, ...],
    required_sources: tuple[str, ...],
    forbidden_terms: tuple[str, ...],
) -> None:
    result = qms_service.search(query, mode="local", limit=8)
    answer = str(result["answer"])
    cited_sources = [
        str(citation.get("doc_id", ""))
        for citation in result.get("citations", [])
        if isinstance(citation, dict)
    ]

    assert result["query_plan"]["intent"] == intent
    for term in required_terms:
        assert term.casefold() in answer.casefold()
    for source in required_sources:
        assert any(source.casefold() in cited.casefold() for cited in cited_sources)
    for term in forbidden_terms:
        assert not _forbidden_present(term, answer)


def _forbidden_present(term: str, answer: str) -> bool:
    normalized_term = term.casefold()
    normalized_answer = answer.casefold()
    if re.fullmatch(r"[a-z0-9]{1,4}", normalized_term):
        return re.search(rf"\b{re.escape(normalized_term)}\b", normalized_answer) is not None
    return normalized_term in normalized_answer
