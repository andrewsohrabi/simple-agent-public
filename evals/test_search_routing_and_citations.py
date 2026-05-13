from __future__ import annotations

import json
import sqlite3

import pytest

from agent.config import SearchConfig
from agent.search.citations import validate_citation_rows, validate_citations
from agent.search.query_plan import plan_query
from agent.search.schema import Citation
from agent.search.service import QmsSearchService
from agent.search.sqlite_store import SearchStore


def _insert_document(
    store: SearchStore,
    *,
    doc_id: str,
    revision: str,
    prefix: str,
    title: str,
    rank: int,
    latest: bool = True,
    obsolete: bool = False,
    signed: bool = False,
    text: str | None = None,
) -> None:
    canonical_doc_key = doc_id
    filename = f"{doc_id}_rev-{revision}{'_signed' if signed else ''}{'_Obsolete' if obsolete else ''}.docx"
    chunk_id = f"{doc_id}:{revision}:metadata:0"
    body = text or f"Document ID: {doc_id}. Revision: {revision}. {title}."
    metadata = {
        "filename": filename,
        "parent_section_id": f"{doc_id}:{revision}:metadata",
        "chunk_index": 0,
    }
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
                canonical_doc_key,
                int(latest),
                int(signed),
                int(obsolete),
                filename,
                filename,
                None,
                f"markdown/{doc_id}_rev-{revision}.md",
                f"sha-{doc_id}-{revision}",
            ),
        )
        conn.execute(
            """
            INSERT INTO revisions
            (canonical_doc_key, doc_id, revision, revision_rank, is_latest, is_obsolete, filename)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (canonical_doc_key, doc_id, revision, rank, int(latest), int(obsolete), filename),
        )
        conn.execute(
            """
            INSERT INTO chunks
            (chunk_id, doc_id, revision, title, section, ordinal, text, search_text,
             parent_section_id, kind, token_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                doc_id,
                revision,
                title,
                "Document Metadata",
                0,
                body,
                body,
                f"{doc_id}:{revision}:metadata",
                "metadata",
                len(body.split()),
                json.dumps(metadata),
            ),
        )
        conn.execute(
            """
            INSERT INTO chunk_fts (chunk_id, doc_id, revision, title, section, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, doc_id, revision, title, "Document Metadata", body),
        )


def _service(tmp_path) -> QmsSearchService:
    config = SearchConfig(
        index_dir=tmp_path,
        openai_vector_store_state=tmp_path / "openai_state.json",
        use_hash_embeddings=True,
        answer_synthesis_enabled=False,
    )
    store = SearchStore(tmp_path / "qms.sqlite")
    store.initialize()
    _insert_document(
        store,
        doc_id="BOM-055",
        revision="F",
        prefix="BOM",
        title="MX1 Bill of Materials",
        rank=6,
        latest=False,
        obsolete=True,
        text="BOM-055 Rev F older bill of materials evidence.",
    )
    _insert_document(
        store,
        doc_id="BOM-055",
        revision="G",
        prefix="BOM",
        title="MX1 Bill of Materials",
        rank=7,
        latest=True,
        text="BOM-055 Rev G latest active bill of materials evidence.",
    )
    _insert_document(
        store,
        doc_id="ECR-593",
        revision="A",
        prefix="ECR",
        title="Engineering Change Request for BOM-055 Rev G",
        rank=1,
        latest=True,
        signed=True,
        text="ECR-593 supports the transition to BOM-055 Rev G. Status: approved.",
    )
    _insert_document(
        store,
        doc_id="VVPR-P01-179",
        revision="B",
        prefix="VVPR",
        title="MX1 Radiographic Verification Protocol",
        rank=2,
        latest=True,
        text="Radiographic verification protocol for MX1 electrical safety.",
    )
    _insert_document(
        store,
        doc_id="RSK-001",
        revision="D",
        prefix="RSK",
        title="MX1 Risk Analysis",
        rank=4,
        latest=True,
        text="Risk analysis includes electrical leakage control traced to verification.",
    )
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO doc_references
            (source_doc_id, source_revision, target_doc_id, reference_text, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("RSK-001", "D", "VVPR-P01-179", "electrical leakage verification", "test"),
        )
    return QmsSearchService(config, use_hash_embeddings=True)


def test_service_routes_counts_lists_and_revision_chains_to_sql(tmp_path):
    service = _service(tmp_path)

    class HostedExplodes:
        def is_available(self):
            return True

        def search(self, query, store, *, limit):
            raise AssertionError("SQL-backed plans must not call hosted search")

    service.hosted_search = HostedExplodes()

    count = service.search("How many BOM revisions are present for BOM-055?", mode="local")
    assert count["query_plan"]["strategy"] == "sql_count"
    assert count["retrieval_backend"] == "sql_inventory"
    assert "Count: 2 BOM-055 document revisions" in count["answer"]
    assert {doc["revision"] for doc in count["retrieved_documents"]} == {"F", "G"}

    listing = service.search("What verification test protocols do we have for the MX1?", mode="local")
    assert listing["query_plan"]["strategy"] == "sql_list"
    assert listing["retrieval_backend"] == "sql_inventory"
    assert listing["retrieved_documents"][0]["doc_id"] == "VVPR-P01-179"

    chain = service.search(
        "What is the latest active revision of BOM-055 and which older revisions exist?",
        mode="local",
    )
    assert chain["query_plan"]["strategy"] == "revision_chain"
    assert chain["retrieval_backend"] == "revision_chain"
    assert "Rev G latest" in chain["answer"]
    assert "Rev F obsolete" in chain["answer"]


def test_unsigned_sql_list_filters_unsigned_instead_of_signed(tmp_path):
    service = _service(tmp_path)

    documents = service._documents_for_sql_plan(
        plan_query("List unsigned QMS documents"),
        limit=10,
    )

    assert documents
    assert all(not document["is_signed"] for document in documents)
    assert {document["doc_id"] for document in documents} >= {"BOM-055", "VVPR-P01-179"}


def test_signed_versus_unsigned_sql_list_keeps_both_signature_states(tmp_path):
    service = _service(tmp_path)

    documents = service._documents_for_sql_plan(
        plan_query("List signed versus unsigned QMS documents"),
        limit=10,
    )

    assert any(document["is_signed"] for document in documents)
    assert any(not document["is_signed"] for document in documents)


def test_scoped_sql_list_applies_filters_to_prefix_and_doc_id_fast_paths(tmp_path):
    service = _service(tmp_path)
    _insert_document(
        service.store,
        doc_id="VVPR-P01-180",
        revision="A",
        prefix="VVPR",
        title="Signed MX1 Verification Protocol",
        rank=1,
        latest=True,
        signed=True,
        text="Signed verification protocol that should not satisfy unsigned scope.",
    )

    unsigned_protocols = service._documents_for_sql_plan(
        plan_query("List unsigned verification protocols"),
        limit=10,
    )
    obsolete_bom = service._documents_for_sql_plan(
        plan_query("List obsolete BOM revisions for BOM-055"),
        limit=10,
    )

    assert [document["doc_id"] for document in unsigned_protocols] == ["VVPR-P01-179"]
    assert all(not document["is_signed"] for document in unsigned_protocols)
    assert [(document["doc_id"], document["revision"]) for document in obsolete_bom] == [
        ("BOM-055", "F")
    ]
    assert all(document["is_obsolete"] for document in obsolete_bom)


def test_non_obsolete_sql_count_uses_active_scope(tmp_path):
    service = _service(tmp_path)

    result = service.search("How many documents are not obsolete?", mode="local")

    assert result["query_plan"]["include_obsolete"] is False
    assert result["retrieval_backend"] == "sql_inventory"
    assert "Count: 4 matching document revisions" in result["answer"]
    assert result["retrieved_documents"]
    assert all(
        not document["metadata"]["is_obsolete"]
        for document in result["retrieved_documents"]
    )
    assert ("BOM-055", "F") not in {
        (document["doc_id"], document["revision"])
        for document in result["retrieved_documents"]
    }


def test_unsigned_title_ranked_documents_do_not_use_signed_filter(tmp_path):
    service = _service(tmp_path)

    documents = service._title_ranked_documents(
        plan_query("Show unsigned verification protocols"),
        limit=10,
    )

    assert [document["doc_id"] for document in documents] == ["VVPR-P01-179"]
    assert all(not document["is_signed"] for document in documents)


def test_service_uses_hosted_search_then_falls_back_to_local(tmp_path):
    service = _service(tmp_path)

    class HostedHit:
        def is_available(self):
            return True

        def search(self, query, store, *, limit):
            return store.chunks_for_documents(
                store.find_documents(doc_id="ECR-593", latest_only=True),
                limit_per_doc=1,
            )

    service.hosted_search = HostedHit()
    hosted = service.search("approved transition change request", mode="auto")
    assert hosted["retrieved_documents"][0]["doc_id"] == "ECR-593"
    assert hosted["retrieval_backend"] == "hosted_file_search"
    assert "hosted_file_search_used" in hosted["warnings"]

    hosted_strict = service.search("approved transition change request", mode="hosted")
    assert hosted_strict["retrieved_documents"][0]["doc_id"] == "ECR-593"
    assert hosted_strict["retrieval_backend"] == "hosted_file_search"

    class HostedMiss:
        def is_available(self):
            return True

        def search(self, query, store, *, limit):
            return []

    service.hosted_search = HostedMiss()
    fallback = service.search("Radiographic verification protocol", mode="auto")
    assert fallback["retrieved_documents"][0]["doc_id"] == "VVPR-P01-179"
    assert fallback["retrieval_backend"] in {"local_hybrid", "local_fts"}
    assert "hosted_file_search_no_hits" in fallback["warnings"]

    hosted_fallback = service.search("Radiographic verification protocol", mode="hosted")
    assert hosted_fallback["retrieved_documents"][0]["doc_id"] == "VVPR-P01-179"
    assert hosted_fallback["retrieval_backend"] in {"local_hybrid", "local_fts"}
    assert "hosted_file_search_no_hits" in hosted_fallback["warnings"]


def test_local_retrieval_handles_fts_failure_after_vector_error(tmp_path, monkeypatch):
    service = _service(tmp_path)

    class HybridExplodes:
        def search_with_trace(self, *args, **kwargs):
            raise RuntimeError("vector index unavailable")

    def fts_raises(*args, **kwargs):
        raise sqlite3.OperationalError("missing schema")

    monkeypatch.setattr(service.vector_index, "exists", lambda: True)
    service.hybrid = HybridExplodes()
    monkeypatch.setattr(service.store, "fts_search", fts_raises)

    result = service.search("unmatched local retrieval query", mode="local")

    assert result["retrieval_backend"] == "local_fts"
    assert result["retrieved_documents"] == []
    assert "vector_search_fallback:RuntimeError" in result["warnings"]
    assert "local_fts_fallback_failed:OperationalError" in result["warnings"]


@pytest.mark.parametrize("mode", ["local", "hybrid"])
def test_local_and_hybrid_modes_never_call_hosted_search(tmp_path, mode):
    service = _service(tmp_path)

    class HostedExplodes:
        def is_available(self):
            return True

        def search(self, query, store, *, limit):
            raise AssertionError(f"{mode} mode must not call hosted search")

    service.hosted_search = HostedExplodes()

    result = service.search("older bill of materials evidence", mode=mode)

    assert result["retrieval_backend"] in {"local_hybrid", "local_fts"}
    assert result["retrieved_documents"][0]["doc_id"] == "BOM-055"
    assert "hosted_file_search_used" not in result["warnings"]


@pytest.mark.parametrize("mode", ["local", "hybrid"])
def test_default_hybrid_retrieval_excludes_obsolete_hits_and_citations(tmp_path, mode):
    service = _service(tmp_path)

    result = service.search("older bill of materials evidence", mode=mode)

    assert result["query_plan"]["strategy"] == "hybrid"
    assert result["retrieved_documents"]
    assert all(
        not doc["metadata"].get("is_obsolete")
        for doc in result["retrieved_documents"]
    )
    assert {
        (doc["doc_id"], doc["revision"])
        for doc in result["retrieved_documents"]
    } == {("BOM-055", "G")}
    assert validate_citation_rows(service.store, result["citations"]) == []
    assert {
        (citation["doc_id"], citation["revision"])
        for citation in result["citations"]
    } == {("BOM-055", "G")}


def test_explicit_obsolete_query_includes_obsolete_records(tmp_path):
    service = _service(tmp_path)

    result = service.search("obsolete older bill of materials evidence", mode="local")

    assert result["query_plan"]["include_obsolete"] is True
    assert result["retrieved_documents"]
    assert all(
        doc["metadata"].get("is_obsolete")
        for doc in result["retrieved_documents"]
    )
    assert {
        (doc["doc_id"], doc["revision"])
        for doc in result["retrieved_documents"]
    } == {("BOM-055", "F")}
    assert validate_citation_rows(service.store, result["citations"]) == []


def test_obsolete_multi_hop_reference_expansion_keeps_obsolete_targets(tmp_path):
    service = _service(tmp_path)
    _insert_document(
        service.store,
        doc_id="VVPR-P01-179",
        revision="A",
        prefix="VVPR",
        title="Legacy Verification Protocol",
        rank=1,
        latest=False,
        obsolete=True,
        text="Legacy target evidence linked only by reference.",
    )

    result = service.search(
        "Trace obsolete evidence from the risk analysis through references.",
        mode="local",
    )

    assert result["query_plan"]["category"] == "traceability"
    assert result["retrieved_documents"]
    assert all(
        doc["metadata"].get("is_obsolete")
        for doc in result["retrieved_documents"]
    )
    assert ("VVPR-P01-179", "A") in {
        (doc["doc_id"], doc["revision"])
        for doc in result["retrieved_documents"]
    }


def test_multi_hop_traceability_follows_local_references(tmp_path):
    service = _service(tmp_path)
    result = service.search(
        "Trace the requirement for electrical leakage testing from the risk file through to the verification report.",
        mode="local",
    )
    doc_ids = {doc["doc_id"] for doc in result["retrieved_documents"]}
    assert result["query_plan"]["category"] == "traceability"
    assert {"RSK-001", "VVPR-P01-179"}.issubset(doc_ids)


def test_citation_validator_checks_chunks_and_metadata_rows(tmp_path):
    service = _service(tmp_path)
    store = service.store
    valid = Citation(
        doc_id="BOM-055",
        revision="G",
        title="MX1 Bill of Materials",
        section="Document Metadata",
        filename="BOM-055_rev-G.docx",
        chunk_id="BOM-055:G:metadata:0",
    )
    assert validate_citations(store, [valid]) == []

    mismatched = Citation(
        doc_id="BOM-055",
        revision="F",
        title="MX1 Bill of Materials",
        section="Document Metadata",
        filename="BOM-055_rev-F.docx",
        chunk_id="BOM-055:G:metadata:0",
    )
    assert "metadata mismatch" in validate_citations(store, [mismatched])[0]

    errors = validate_citation_rows(
        store,
        [
            {
                "doc_id": "MISSING-001",
                "revision": "A",
                "title": "Missing",
                "section": "metadata_inventory",
                "filename": "missing.docx",
            }
        ],
    )
    assert errors == ["Unknown citation document MISSING-001 Rev A"]
