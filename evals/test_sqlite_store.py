from pathlib import Path

from agent.config import SearchConfig
from agent.search.ingest import ingest_corpus
from agent.search.sqlite_store import SearchStore


def _manifest(tmp_path):
    markdown_path = tmp_path / "BOM-055.md"
    markdown_path.write_text(
        "# MX1 BOM\n\nBOM-055 Rev G references ECR-123 for release approval.",
        encoding="utf-8",
    )
    return {
        "created_at": "2026-05-07T00:00:00Z",
        "source_zip": "sample.zip",
        "source_sha256": "abc123",
        "document_count": 1,
        "skipped_empty_count": 0,
        "documents": [
            {
                "doc_id": "BOM-055",
                "revision": "G",
                "prefix": "BOM",
                "title": "MX1 Bill of Materials",
                "revision_rank": 7,
                "canonical_doc_key": "BOM-055",
                "is_latest": True,
                "is_signed": True,
                "is_obsolete": False,
                "filename": "BOM-055.docx",
                "source_path": "qms/BOM-055.docx",
                "software_version": None,
                "markdown_path": str(markdown_path),
                "sha256": "doc-sha",
            }
        ],
    }


def test_sqlite_store_populates_index_tables_from_manifest(tmp_path):
    store = SearchStore(tmp_path / "qms.sqlite")
    chunks = store.load_manifest(
        _manifest(tmp_path),
        config=SearchConfig(
            min_chunk_tokens=1,
            chunk_size_tokens=20,
            max_chunk_tokens=40,
            index_dir=tmp_path,
        ),
    )

    assert chunks
    assert store.stats() == {
        "exists": True,
        "documents": 1,
        "chunks": len(chunks),
        "latest_documents": 1,
        "references": 1,
        "schema_current": True,
        "requires_rebuild": False,
    }
    assert store.revision_chain(doc_id="BOM-055")[0]["revision"] == "G"
    assert store.references_from("BOM-055", "G")[0]["target_doc_id"] == "ECR-123"
    assert store.fts_search("release approval", limit=1)
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM ingest_runs").fetchone()[0] == 1


def test_sqlite_store_extracts_compact_3p_references(tmp_path):
    manifest = _manifest(tmp_path)
    markdown_path = tmp_path / "MEMO-P01-685.md"
    markdown_path.write_text(
        "# Summary\n\nElectrical Safety Testing references 3P-P01-33 and EMC references 3P-P01-32.",
        encoding="utf-8",
    )
    manifest["documents"] = [
        {
            **manifest["documents"][0],
            "doc_id": "MEMO-P01-685",
            "prefix": "MEMO",
            "title": "MX1 Design Verification and Validation Summary",
            "filename": "MEMO-P01-685.docx",
            "source_path": "qms/MEMO-P01-685.docx",
            "markdown_path": str(markdown_path),
        }
    ]
    store = SearchStore(tmp_path / "qms.sqlite")
    store.load_manifest(manifest)

    references = {row["target_doc_id"] for row in store.references_from("MEMO-P01-685", "G")}
    assert {"3P-P01-33", "3P-P01-32"} <= references


def test_sqlite_schema_current_requires_fts_table(tmp_path):
    store = SearchStore(tmp_path / "qms.sqlite")
    store.initialize()
    assert store.schema_current() is True

    with store.connect() as conn:
        conn.execute("DROP TABLE chunk_fts")

    assert store.schema_current() is False
    assert store.stats()["requires_rebuild"] is True


def test_sqlite_store_supports_counts_and_citations(tmp_path):
    zip_path = Path("Example_QMS_-_MedAI.zip")
    if not zip_path.exists():
        return
    manifest = ingest_corpus(zip_path, tmp_path)
    store = SearchStore(tmp_path / "qms.sqlite")
    chunks = store.load_manifest(manifest)
    assert store.stats()["documents"] == manifest["document_count"]
    assert store.stats()["references"] >= 1
    assert chunks

    bom = store.count_by_prefix("BOM")
    assert bom["count"] >= 1
    hits = store.fts_search("Bill of Materials MX1", limit=5)
    assert hits
    assert store.citation_for_chunk(hits[0].chunk_id) is not None
    chain = store.revision_chain(prefix="BOM", include_obsolete=True)
    assert chain


def test_sqlite_store_reports_schema_current_after_load(tmp_path):
    zip_path = Path("Example_QMS_-_MedAI.zip")
    if not zip_path.exists():
        return
    manifest = ingest_corpus(zip_path, tmp_path / "index")
    store = SearchStore(tmp_path / "qms.sqlite")
    store.load_manifest(manifest)
    stats = store.stats()
    assert stats["schema_current"] is True
    assert stats["requires_rebuild"] is False
