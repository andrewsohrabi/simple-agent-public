import io
import zipfile
from pathlib import Path

from agent.config import SearchConfig
from agent.search.chunking import chunks_from_manifest
from agent.search.docx_extract import extract_docx
from agent.search.ingest import ingest_corpus
from agent.search.index_contracts import (
    ARTIFACT_CONTRACT_VERSION,
    INGEST_MANIFEST_ARTIFACT_TYPE,
    INGEST_MANIFEST_SCHEMA_VERSION,
)
from agent.search.metadata import (
    document_family,
    parse_document_metadata,
    project_code_from_filename,
)


def _docx_bytes(document_xml: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


def _document_xml(body: str) -> str:
    return (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )


def _paragraph(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def _table(rows: list[list[str]], *, header: bool = False) -> str:
    row_xml = []
    for index, row in enumerate(rows):
        cells = "".join(f"<w:tc>{_paragraph(cell)}</w:tc>" for cell in row)
        props = "<w:trPr><w:tblHeader/></w:trPr>" if header and index == 0 else ""
        row_xml.append(f"<w:tr>{props}{cells}</w:tr>")
    return f"<w:tbl>{''.join(row_xml)}</w:tbl>"


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


def test_filename_helpers_extract_family_and_project_code():
    filename = "VVPR-P01-179 - MX1 Software System v3.0.0 Protocol and Report_B.docx"
    assert document_family("VVPR-P01-179") == "VVPR"
    assert project_code_from_filename(filename, "VVPR-P01-179") == "P01"
    assert project_code_from_filename("MEMO-P01-655 - P01-MAI UI Spec_C.docx") == "P01"


def test_extract_docx_reads_paragraphs_and_tables():
    data = _docx_bytes(
        _document_xml(
            _paragraph("Protocol introduction")
            + _table(
                [
                    ["Step", "Expected"],
                    ["1", "System starts"],
                    ["2", "Alarm remains clear"],
                ]
            )
            + _paragraph("Protocol conclusion")
        )
    )

    extracted = extract_docx(data)

    assert extracted.warnings == []
    assert extracted.paragraphs == ["Protocol introduction", "Protocol conclusion"]
    assert extracted.tables == [
        [
            ["Step", "Expected"],
            ["1", "System starts"],
            ["2", "Alarm remains clear"],
        ]
    ]
    assert "Step | Expected" in extracted.plain_text


def test_ingest_preserves_docx_paragraph_table_order(tmp_path):
    zip_path = tmp_path / "corpus.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "Example QMS - MedAI/MEMO-P01-999 - Ordered Content_A.docx",
            _docx_bytes(
                _document_xml(
                    _paragraph("Intro before the table")
                    + _table([["Step", "Expected"], ["1", "System starts"]], header=True)
                    + _paragraph("Conclusion after the table")
                )
            ),
        )

    manifest = ingest_corpus(zip_path, tmp_path / "index")
    markdown = Path(str(manifest["documents"][0]["markdown_path"])).read_text(
        encoding="utf-8"
    )

    assert markdown.index("Intro before the table") < markdown.index("### Table 1")
    assert markdown.index("### Table 1") < markdown.index("Conclusion after the table")


def test_headerless_docx_table_keeps_first_row_as_data(tmp_path):
    zip_path = tmp_path / "corpus.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "Example QMS - MedAI/MEMO-P01-998 - Headerless Table_A.docx",
            _docx_bytes(
                _document_xml(
                    _table([["R1C1", "R1C2"], ["R2C1", "R2C2"]], header=False)
                )
            ),
        )

    manifest = ingest_corpus(zip_path, tmp_path / "index")
    chunks = chunks_from_manifest(
        manifest,
        config=SearchConfig(create_metadata_chunks=False),
    )
    table_rows = [chunk for chunk in chunks if chunk.kind == "table_row"]

    assert [chunk.row_start for chunk in table_rows] == [1, 2]
    assert "Column 1=R1C1" in table_rows[0].text
    assert "Column 2=R1C2" in table_rows[0].text


def test_ingest_corpus_records_metadata_only_warning_for_empty_docx(tmp_path):
    zip_path = tmp_path / "corpus.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "Example QMS - MedAI/BOM-055 - MX1 Top-level assembly_G.docx",
            _docx_bytes(_document_xml("")),
        )

    manifest = ingest_corpus(zip_path, tmp_path / "index")

    assert manifest["document_count"] == 1
    assert manifest["artifact_type"] == INGEST_MANIFEST_ARTIFACT_TYPE
    assert manifest["artifact_contract_version"] == ARTIFACT_CONTRACT_VERSION
    assert manifest["schema_version"] == INGEST_MANIFEST_SCHEMA_VERSION
    assert manifest["skipped_empty_count"] == 1
    assert manifest["metadata_only_count"] == 1
    assert manifest["documents"][0]["warnings"] == ["empty_body_metadata_only"]
    assert manifest["documents"][0]["family"] == "BOM"
    assert (
        manifest["documents"][0]["source_hash"]
        == manifest["documents"][0]["sha256"]
    )
    assert manifest["documents"][0]["latest_non_obsolete"] is True
    markdown = Path(str(manifest["documents"][0]["markdown_path"])).read_text(
        encoding="utf-8"
    )
    assert "- Family: BOM" in markdown
    assert "- Source hash: " in markdown
    assert "- Extraction warnings: empty_body_metadata_only" in markdown


def test_ingest_corpus_extracts_real_docx(tmp_path):
    zip_path = Path("Example_QMS_-_MedAI.zip")
    if not zip_path.exists():
        return
    manifest = ingest_corpus(zip_path, tmp_path)
    assert manifest["document_count"] == 189
    assert manifest["metadata_only_count"] > 0
    assert manifest["skipped_empty_count"] == manifest["metadata_only_count"]
    assert Path(manifest["normalized_dir"]).exists()
    assert any(doc["doc_id"].startswith("BOM") for doc in manifest["documents"])
    assert all("__MACOSX/" not in doc["source_path"] for doc in manifest["documents"])
    assert all(
        not Path(doc["source_path"]).name.startswith("._")
        for doc in manifest["documents"]
    )

    bom = next(
        doc
        for doc in manifest["documents"]
        if doc["doc_id"] == "BOM-055" and doc["revision"] == "G"
    )
    assert bom["family"] == "BOM"
    assert bom["project_code"] is None
    assert bom["source_filename"] == bom["filename"]
    assert bom["source_hash"] == bom["sha256"]
    assert bom["latest_non_obsolete"] is True
    assert len(bom["source_hash"]) == 64

    obsolete = next(doc for doc in manifest["documents"] if doc["is_obsolete"])
    assert obsolete["latest_non_obsolete"] is False
