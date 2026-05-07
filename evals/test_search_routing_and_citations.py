from __future__ import annotations

import json

import pytest

from agent.config import SearchConfig
from agent.search.citations import validate_citation_rows, validate_citations
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
    assert any(
        doc["doc_id"] == "BOM-055"
        and doc["revision"] == "F"
        and doc["metadata"].get("is_obsolete")
        for doc in result["retrieved_documents"]
    )
    assert validate_citation_rows(service.store, result["citations"]) == []


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
