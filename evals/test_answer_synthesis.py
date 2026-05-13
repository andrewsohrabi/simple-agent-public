from __future__ import annotations

from datetime import date as real_date
import json

import agent.search.answer as answer_module
from agent.config import SearchConfig
from agent.search.answer import SearchAnswerer
from agent.search.query_plan import QueryPlan, plan_query
from agent.search.schema import SearchHit
from agent.search.sqlite_store import SearchStore


def _store(tmp_path) -> SearchStore:
    store = SearchStore(tmp_path / "qms.sqlite")
    store.initialize()
    body = (
        "BOM-055 Rev G is the latest active bill of materials. "
        "It lists MX1 assembly material controls and approved component records."
    )
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO documents
            (doc_id, revision, prefix, title, revision_rank, canonical_doc_key,
             is_latest, is_signed, is_obsolete, filename, source_path, software_version,
             markdown_path, sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BOM-055",
                "G",
                "BOM",
                "MX1 Bill of Materials",
                7,
                "BOM-055",
                1,
                1,
                0,
                "BOM-055_rev-G.docx",
                "BOM-055_rev-G.docx",
                None,
                "markdown/BOM-055_rev-G.md",
                "sha-BOM-055-G",
            ),
        )
        conn.execute(
            """
            INSERT INTO revisions
            (canonical_doc_key, doc_id, revision, revision_rank, is_latest, is_obsolete, filename)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("BOM-055", "BOM-055", "G", 7, 1, 0, "BOM-055_rev-G.docx"),
        )
        conn.execute(
            """
            INSERT INTO chunks
            (chunk_id, doc_id, revision, title, section, ordinal, text, search_text,
             parent_section_id, kind, token_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BOM-055:G:metadata:0",
                "BOM-055",
                "G",
                "MX1 Bill of Materials",
                "Document Metadata",
                0,
                body,
                body,
                "BOM-055:G:metadata",
                "metadata",
                len(body.split()),
                json.dumps({"filename": "BOM-055_rev-G.docx"}),
            ),
        )
    return store


def _insert_metadata_document(
    store: SearchStore,
    *,
    doc_id: str,
    revision: str = "A",
    prefix: str,
    title: str,
    rank: int = 1,
    latest: bool = True,
    signed: bool = False,
    obsolete: bool = False,
    markdown_path=None,
) -> None:
    filename = f"{doc_id}_rev-{revision}.docx"
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO documents
            (doc_id, revision, prefix, title, revision_rank, canonical_doc_key,
             is_latest, is_signed, is_obsolete, filename, source_path, software_version,
             markdown_path, sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                revision,
                prefix,
                title,
                rank,
                doc_id,
                int(latest),
                int(signed),
                int(obsolete),
                filename,
                filename,
                None,
                str(markdown_path or f"markdown/{doc_id}_rev-{revision}.md"),
                f"sha-{doc_id}-{revision}",
            ),
        )
        conn.execute(
            """
            INSERT INTO revisions
            (canonical_doc_key, doc_id, revision, revision_rank, is_latest, is_obsolete, filename)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, doc_id, revision, rank, int(latest), int(obsolete), filename),
        )


def _insert_ecr_document(
    store: SearchStore,
    tmp_path,
    *,
    doc_id: str,
    dco: str,
    approval_date: str,
) -> None:
    markdown_path = tmp_path / f"{doc_id}.md"
    markdown_path.write_text(
        "\n".join(
            [
                f"# {doc_id}",
                "Affected document: BOM-055.",
                "",
                "DOCUMENT APPROVALS",
                "",
                f"| | A | {dco} | Approved | QA | {approval_date} |",
            ]
        ),
        encoding="utf-8",
    )
    _insert_metadata_document(
        store,
        doc_id=doc_id,
        prefix="ECR",
        title=f"Engineering Change Request {doc_id}",
        signed=True,
        markdown_path=markdown_path,
    )


def _freeze_answer_date(monkeypatch, year: int = 2026, month: int = 5, day: int = 13):
    class FrozenDate(real_date):
        @classmethod
        def today(cls):
            return cls(year, month, day)

    monkeypatch.setattr(answer_module, "date", FrozenDate)


def _hit() -> SearchHit:
    return SearchHit(
        chunk_id="BOM-055:G:metadata:0",
        doc_id="BOM-055",
        revision="G",
        title="MX1 Bill of Materials",
        section="Document Metadata",
        text="BOM-055 Rev G is the latest active bill of materials for MX1.",
        score=0.98,
        source="hybrid",
        metadata={"is_obsolete": False},
    )


def _revision_hit(revision: str) -> SearchHit:
    return SearchHit(
        chunk_id=f"BOM-055:{revision}:metadata:0",
        doc_id="BOM-055",
        revision=revision,
        title="MX1 Bill of Materials",
        section="Document Metadata",
        text=f"BOM-055 Rev {revision} bill of materials evidence.",
        score=1.0,
        source="metadata",
        metadata={"is_obsolete": False},
    )


def _plan(**overrides) -> QueryPlan:
    values = {
        "category": "known_item",
        "strategy": "hybrid",
        "query": "Summarize BOM-055",
    }
    values.update(overrides)
    return QueryPlan(**values)


class RecordingSynthesizer:
    def __init__(self, text: str | None = None, error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls: list[dict[str, object]] = []

    def synthesize(self, *, query, hits, citation_rows, model, max_input_chars):
        self.calls.append(
            {
                "query": query,
                "hits": hits,
                "citation_rows": citation_rows,
                "model": model,
                "max_input_chars": max_input_chars,
            }
        )
        if self.error:
            raise self.error
        return self.text


def test_synthesized_answer_uses_config_chat_model_and_keeps_local_citations(tmp_path):
    store = _store(tmp_path)
    synthesizer = RecordingSynthesizer(
        "BOM-055 Rev G is the active bill of materials for MX1. [S1]"
    )
    answerer = SearchAnswerer(store, config=SearchConfig(), synthesizer=synthesizer)

    result = answerer.answer("Summarize BOM-055", _plan(), [_hit()])

    assert result["answer"] == "BOM-055 Rev G is the active bill of materials for MX1. [S1]"
    citation = result["citations"][0]
    expected_citation_fields = {
        "doc_id": "BOM-055",
        "revision": "G",
        "title": "MX1 Bill of Materials",
        "section": "Document Metadata",
        "filename": "BOM-055_rev-G.docx",
        "markdown_path": "markdown/BOM-055_rev-G.md",
        "chunk_id": "BOM-055:G:metadata:0",
    }
    assert expected_citation_fields.items() <= citation.items()
    assert citation["markdown_path_abs"].endswith("markdown/BOM-055_rev-G.md")
    assert citation["source_path_abs"].endswith("BOM-055_rev-G.docx")
    assert result["warnings"] == []
    assert synthesizer.calls[0]["model"] == "gpt-5.5"
    assert synthesizer.calls[0]["max_input_chars"] == 12000


def test_synthesis_failure_falls_back_to_deterministic_answer_with_warning(tmp_path):
    store = _store(tmp_path)
    synthesizer = RecordingSynthesizer(error=RuntimeError("network unavailable"))
    answerer = SearchAnswerer(store, config=SearchConfig(), synthesizer=synthesizer)

    result = answerer.answer("Summarize BOM-055", _plan(), [_hit()])

    assert "I found source-backed MedAI QMS evidence" in result["answer"]
    assert "BOM-055 Rev G, Document Metadata" in result["answer"]
    assert "answer_synthesis_fallback:RuntimeError" in result["warnings"]
    assert result["citations"][0]["doc_id"] == "BOM-055"


def test_revision_diff_without_explicit_pair_uses_latest_first_hit_order(tmp_path):
    store = _store(tmp_path)
    answerer = SearchAnswerer(
        store,
        config=SearchConfig(answer_synthesis_enabled=False),
    )

    result = answerer.answer(
        "what changed in the latest version from previous versions?",
        _plan(strategy="revision_diff", requires_diff=True),
        [_revision_hit("G"), _revision_hit("F"), _revision_hit("E")],
    )

    assert "Requested comparison: G vs F." in result["answer"]
    assert "- Rev G:" in result["answer"]
    assert "- Rev F:" in result["answer"]


def test_count_answer_path_avoids_answer_synthesis(tmp_path):
    store = _store(tmp_path)
    synthesizer = RecordingSynthesizer("Should not be used. [S1]")
    answerer = SearchAnswerer(store, config=SearchConfig(), synthesizer=synthesizer)

    result = answerer.answer(
        "How many BOM-055 records are present?",
        _plan(strategy="sql_count", requires_count=True, doc_id="BOM-055"),
        [_hit()],
    )

    assert result["answer"].startswith("Count: 1 BOM-055 document revisions.")
    assert synthesizer.calls == []


def test_count_answer_uses_full_metadata_count_not_sample_limit(tmp_path):
    store = SearchStore(tmp_path / "qms.sqlite")
    store.initialize()
    with store.connect() as conn:
        for index in range(501):
            doc_id = f"DOC-{index:03}"
            conn.execute(
                """
                INSERT INTO documents
                (doc_id, revision, prefix, title, revision_rank, canonical_doc_key,
                 is_latest, is_signed, is_obsolete, filename, source_path, software_version,
                 markdown_path, sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    "A",
                    "DOC",
                    f"Document {index}",
                    1,
                    doc_id,
                    1,
                    0,
                    0,
                    f"{doc_id}.docx",
                    f"qms/{doc_id}.docx",
                    None,
                    f"markdown/{doc_id}.md",
                    f"sha-{doc_id}",
                ),
            )
    answerer = SearchAnswerer(store, config=SearchConfig(), synthesizer=None)

    result = answerer.answer(
        "How many DOC records are present?",
        _plan(
            category="enumeration",
            strategy="sql_count",
            query="How many DOC records are present?",
            prefix="DOC",
            requires_count=True,
        ),
        [],
    )

    assert result["answer"].startswith("Count: 501 DOC document revisions.")
    assert len(result["retrieved_documents"]) == 40


def test_scoped_count_answer_applies_signature_and_obsolete_filters(tmp_path):
    store = SearchStore(tmp_path / "qms.sqlite")
    store.initialize()
    _insert_metadata_document(
        store,
        doc_id="VVPR-P01-179",
        prefix="VVPR",
        title="Unsigned Verification Protocol",
        signed=False,
    )
    _insert_metadata_document(
        store,
        doc_id="VVPR-P01-180",
        prefix="VVPR",
        title="Signed Verification Protocol",
        signed=True,
    )
    _insert_metadata_document(
        store,
        doc_id="BOM-055",
        revision="F",
        prefix="BOM",
        title="Obsolete BOM",
        rank=6,
        latest=False,
        obsolete=True,
    )
    _insert_metadata_document(
        store,
        doc_id="BOM-055",
        revision="G",
        prefix="BOM",
        title="Active BOM",
        rank=7,
        latest=True,
    )
    answerer = SearchAnswerer(store, config=SearchConfig(), synthesizer=None)

    unsigned = answerer.answer(
        "How many unsigned verification protocols are present?",
        plan_query("How many unsigned verification protocols are present?"),
        [],
    )
    obsolete = answerer.answer(
        "How many obsolete BOM revisions are present for BOM-055?",
        plan_query("How many obsolete BOM revisions are present for BOM-055?"),
        [],
    )

    assert unsigned["answer"].startswith("Count: 1 VVPR document revisions.")
    assert [doc["doc_id"] for doc in unsigned["retrieved_documents"]] == [
        "VVPR-P01-179"
    ]
    assert all(
        not doc["metadata"]["is_signed"] for doc in unsigned["retrieved_documents"]
    )
    assert obsolete["answer"].startswith("Count: 1 BOM-055 document revisions.")
    assert [(doc["doc_id"], doc["revision"]) for doc in obsolete["retrieved_documents"]] == [
        ("BOM-055", "F")
    ]
    assert all(doc["metadata"]["is_obsolete"] for doc in obsolete["retrieved_documents"])


def test_ecr_status_honors_requested_filing_year(tmp_path, monkeypatch):
    _freeze_answer_date(monkeypatch)
    store = SearchStore(tmp_path / "qms.sqlite")
    store.initialize()
    _insert_ecr_document(
        store,
        tmp_path,
        doc_id="ECR-231",
        dco="23-101",
        approval_date="2023-05-01",
    )
    _insert_ecr_document(
        store,
        tmp_path,
        doc_id="ECR-241",
        dco="24-101",
        approval_date="2024-05-01",
    )
    answerer = SearchAnswerer(store, config=SearchConfig(), synthesizer=None)
    query = "Which ECRs were filed in 2023?"

    result = answerer.answer(query, plan_query(query), [])

    assert "2023-01-01 to 2023-12-31" in result["answer"]
    assert "ECR-231" in result["answer"]
    assert "ECR-241" not in result["answer"]


def test_ecr_status_parses_non_2024_dco_approval_rows(tmp_path, monkeypatch):
    _freeze_answer_date(monkeypatch)
    store = SearchStore(tmp_path / "qms.sqlite")
    store.initialize()
    _insert_ecr_document(
        store,
        tmp_path,
        doc_id="ECR-251",
        dco="25-123",
        approval_date="2025-06-01",
    )
    answerer = SearchAnswerer(store, config=SearchConfig(), synthesizer=None)
    query = "Show me all ECRs filed in the last year and their status"

    result = answerer.answer(query, plan_query(query), [])

    assert "DCO 25-123" in result["answer"]
    assert "approval effective date 2025-06-01" in result["answer"]


def test_ingest_manifest_count_reports_content_bearing_metric(tmp_path):
    store = _store(tmp_path)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "ingest_manifest.json").write_text(
        json.dumps(
            {
                "document_count": 10,
                "skipped_empty_count": 2,
                "metadata_only_count": 2,
            }
        ),
        encoding="utf-8",
    )
    answerer = SearchAnswerer(store, config=SearchConfig(index_dir=index_dir))

    result = answerer.answer(
        "How many non-empty DOCX records were ingested after ignoring empty documents?",
        _plan(
            category="enumeration",
            strategy="sql_count",
            query="How many non-empty DOCX records were ingested after ignoring empty documents?",
            requires_count=True,
            intent="ingest_manifest_count",
        ),
        [],
    )

    assert "Count: 8 non-empty content-bearing DOCX records ingested" in result["answer"]
    assert "Total normalized DOCX records: 10" in result["answer"]
    assert "10 non-empty DOCX records" not in result["answer"]
