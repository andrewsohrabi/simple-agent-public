from pathlib import Path

from agent.search.ingest import ingest_corpus
from agent.search.metadata import parse_document_metadata


def test_parse_document_metadata_handles_revision_status():
    metadata = parse_document_metadata(
        "VVPR-P01-214 - MX1 Software System Beam Angle Accuracy Protocol and Report_C-Obsolete.docx"
    )
    assert metadata.doc_id == "VVPR-P01-214"
    assert metadata.revision == "C"
    assert metadata.prefix == "VVPR"
    assert metadata.is_obsolete is True


def test_parse_document_metadata_prefers_file_revision_suffix_over_title_rev():
    metadata = parse_document_metadata("ECR-593 - MX1 BOM-055 Rev G_A-signed.docx")
    assert metadata.doc_id == "ECR-593"
    assert metadata.revision == "A"
    assert "Rev G" in metadata.title


def test_parse_document_metadata_handles_product_style_ifu_id():
    metadata = parse_document_metadata("IFU-MX1 - MX1 Instructions for Use_L.docx")
    assert metadata.doc_id == "IFU-MX1"
    assert metadata.prefix == "IFU"
    assert metadata.revision == "L"
    assert metadata.title == "MX1 Instructions for Use"


def test_ingest_corpus_extracts_real_docx(tmp_path):
    zip_path = Path("Example_QMS_-_MedAI.zip")
    if not zip_path.exists():
        return
    manifest = ingest_corpus(zip_path, tmp_path)
    assert manifest["document_count"] == 189
    assert manifest["skipped_empty_count"] == 0
    assert manifest["metadata_only_count"] > 0
    assert Path(manifest["normalized_dir"]).exists()
    assert any(doc["doc_id"].startswith("BOM") for doc in manifest["documents"])
