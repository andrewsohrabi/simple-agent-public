from pathlib import Path

from agent.search.ingest import ingest_corpus
from agent.search.sqlite_store import SearchStore


def test_sqlite_store_supports_counts_and_citations(tmp_path):
    zip_path = Path("Example_QMS_-_MedAI.zip")
    if not zip_path.exists():
        return
    manifest = ingest_corpus(zip_path, tmp_path)
    store = SearchStore(tmp_path / "qms.sqlite")
    chunks = store.load_manifest(manifest)
    assert store.stats()["documents"] == manifest["document_count"]
    assert chunks

    bom = store.count_by_prefix("BOM")
    assert bom["count"] >= 1
    hits = store.fts_search("Bill of Materials MX1", limit=5)
    assert hits
    assert store.citation_for_chunk(hits[0].chunk_id) is not None


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
