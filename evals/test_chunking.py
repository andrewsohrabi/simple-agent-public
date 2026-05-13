from pathlib import Path

from agent.config import SearchConfig
from agent.search.chunking import chunk_text, count_tokens
from agent.search.schema import DocumentMetadata
from agent.search.sqlite_store import SearchStore


def metadata(doc_id: str = "VVPR-P01-229") -> DocumentMetadata:
    return DocumentMetadata(
        doc_id=doc_id,
        prefix=doc_id.split("-", 1)[0],
        title="MX1 Software System v3.3.0 Protocol and Report",
        revision="B",
        revision_rank=2,
        canonical_doc_key=doc_id,
        is_latest=True,
        is_signed=False,
        is_obsolete=False,
        filename=f"{doc_id} - MX1 Software System v3.3.0 Protocol and Report_B.docx",
        source_path=f"Example QMS - MedAI/{doc_id}.docx",
        software_version="v3.3.0",
    )


def test_metadata_chunk_is_created_for_every_document():
    chunks = chunk_text(metadata("BOM-055"), "", warnings=["empty_body_metadata_only"])
    assert len(chunks) == 1
    assert chunks[0].kind == "metadata"
    assert "BOM-055" in chunks[0].text
    assert "Filename:" in chunks[0].text


def test_search_text_has_metadata_preamble_but_raw_text_is_clean():
    chunks = chunk_text(metadata(), "# Results\nThe test passed.")
    prose = next(chunk for chunk in chunks if chunk.kind == "prose")
    assert prose.search_text.startswith("Document: VVPR-P01-229")
    assert "Content:\nThe test passed." in prose.search_text
    assert prose.text == "The test passed."


def test_prose_chunks_stay_within_max_chunk_tokens():
    body = " ".join(f"word{i}" for i in range(2500))
    chunks = chunk_text(metadata(), f"# Long Section\n{body}")
    prose_chunks = [chunk for chunk in chunks if chunk.kind == "prose"]
    assert prose_chunks
    assert all(chunk.token_count <= 900 for chunk in prose_chunks)


def test_short_sections_are_merged_unless_metadata_important():
    markdown = "# Tiny\nShort note.\n# Main\n" + " ".join("content" for _ in range(160))
    chunks = chunk_text(metadata(), markdown)
    prose = [chunk for chunk in chunks if chunk.kind == "prose"]
    assert len(prose) == 1
    assert "Tiny" in prose[0].text
    assert "Short note" in prose[0].text


def test_table_rows_are_not_split_and_headers_repeat():
    rows = "\n".join(f"| TC-{i:03d} | REQ-{i:03d} | Pass |" for i in range(1, 80))
    table = "\n".join(
        [
            "# Verification Results",
            "| Test ID | Requirement | Pass/Fail |",
            "| --- | --- | --- |",
            rows,
        ]
    )
    config = SearchConfig(table_chunk_target_tokens=80, table_chunk_max_tokens=100)
    chunks = chunk_text(metadata(), table, config=config)
    table_chunks = [chunk for chunk in chunks if chunk.kind == "table"]
    assert len(table_chunks) > 1
    assert all("Columns: Test ID | Requirement | Pass/Fail" in chunk.text for chunk in table_chunks)
    assert all("Row " in chunk.text for chunk in table_chunks)
    assert not any("TC-001 | REQ-001" in chunk.text and "TC-079 | REQ-079" in chunk.text for chunk in table_chunks)


def test_chunk_ids_are_stable_across_repeated_indexing():
    markdown = "# Results\nThe verification protocol passed."
    first = chunk_text(metadata(), markdown)
    second = chunk_text(metadata(), markdown)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunks_include_citation_metadata():
    chunks = chunk_text(metadata(), "# Results\nThe verification protocol passed.")
    prose = next(chunk for chunk in chunks if chunk.kind == "prose")

    assert prose.chunk_id
    assert prose.metadata["filename"] == metadata().filename
    assert prose.metadata["source_path"] == metadata().source_path
    assert prose.metadata["document_code"] == metadata().doc_id
    assert prose.metadata["parent_section_id"] == prose.parent_section_id
    assert prose.metadata["heading_path"] == ["Results"]
    assert prose.metadata["chunk_index"] == prose.chunk_index
    assert prose.metadata["ordinal_start"] == prose.ordinal_start
    assert prose.metadata["ordinal_end"] == prose.ordinal_end


def test_metadata_only_document_is_retrievable_by_code_and_filename(tmp_path):
    doc_path = tmp_path / "IFU-MX1_rev-D.md"
    doc_path.write_text("# IFU-MX1 Rev D: MX1 Instructions for Use\n", encoding="utf-8")
    manifest = {
        "created_at": "now",
        "source_zip": "test.zip",
        "source_sha256": "hash",
        "document_count": 1,
        "skipped_empty_count": 0,
        "documents": [
            {
                "doc_id": "IFU-MX1",
                "prefix": "IFU",
                "title": "MX1 Instructions for Use",
                "revision": "D",
                "revision_rank": 4,
                "canonical_doc_key": "IFU-MX1",
                "is_latest": True,
                "is_signed": False,
                "is_obsolete": False,
                "filename": "IFU-MX1 - MX1 Instructions for Use_D.docx",
                "source_path": "Example QMS - MedAI/IFU-MX1.docx",
                "software_version": None,
                "markdown_path": str(doc_path),
                "sha256": "hash",
                "warnings": ["empty_body_metadata_only"],
            }
        ],
    }
    store = SearchStore(tmp_path / "qms.sqlite")
    store.load_manifest(manifest, config=SearchConfig())
    hits = store.fts_search("IFU-MX1 Instructions for Use", limit=5)
    assert hits
    assert hits[0].doc_id == "IFU-MX1"


def test_selected_answer_chunks_expand_to_neighbors(tmp_path):
    doc_path = tmp_path / "VVPR-P01-229_rev-B.md"
    doc_path.write_text(
        "# Section\n"
        + "\n\n".join(" ".join(f"word{i}_{j}" for j in range(140)) for i in range(4)),
        encoding="utf-8",
    )
    manifest = {
        "created_at": "now",
        "source_zip": "test.zip",
        "source_sha256": "hash",
        "document_count": 1,
        "skipped_empty_count": 0,
        "documents": [
            {
                **metadata().__dict__,
                "markdown_path": str(doc_path),
                "sha256": "hash",
                "warnings": [],
            }
        ],
    }
    store = SearchStore(tmp_path / "qms.sqlite")
    chunks = store.load_manifest(
        manifest,
        config=SearchConfig(chunk_size_tokens=180, chunk_overlap_tokens=30),
    )
    middle = [chunk for chunk in chunks if chunk.kind == "prose"][1]
    hit = store.fts_search(middle.text.split()[0], limit=1)[0]
    expanded = store.expand_neighbors([hit], neighbor_chunks=1, parent_section_max_tokens=1800)
    assert len(expanded) >= 2
    assert {item.metadata["parent_section_id"] for item in expanded} == {
        middle.parent_section_id
    }


def test_prose_chunk_ordinals_advance_when_overlap_is_enabled():
    markdown = "# Section\n" + "\n\n".join(
        " ".join(f"para{index}_{token}" for token in range(60))
        for index in range(5)
    )
    chunks = chunk_text(
        metadata(),
        markdown,
        config=SearchConfig(
            chunk_size_tokens=100,
            chunk_overlap_tokens=30,
            min_chunk_tokens=1,
            max_chunk_tokens=200,
        ),
    )
    prose = [chunk for chunk in chunks if chunk.kind == "prose"]

    assert len(prose) >= 3
    assert prose[-1].ordinal_end == 4
    assert [chunk.ordinal_start for chunk in prose] == sorted(
        chunk.ordinal_start for chunk in prose
    )


def test_neighbor_expansion_honors_parent_section_token_cap(tmp_path):
    doc_path = tmp_path / "VVPR-P01-229_rev-B.md"
    doc_path.write_text(
        "# Section\n"
        + "\n\n".join(" ".join(f"word{i}_{j}" for j in range(140)) for i in range(5)),
        encoding="utf-8",
    )
    manifest = {
        "created_at": "now",
        "source_zip": "test.zip",
        "source_sha256": "hash",
        "document_count": 1,
        "skipped_empty_count": 0,
        "documents": [
            {
                **metadata().__dict__,
                "markdown_path": str(doc_path),
                "sha256": "hash",
                "warnings": [],
            }
        ],
    }
    store = SearchStore(tmp_path / "qms.sqlite")
    chunks = store.load_manifest(
        manifest,
        config=SearchConfig(chunk_size_tokens=180, chunk_overlap_tokens=0),
    )
    middle = [chunk for chunk in chunks if chunk.kind == "prose"][2]
    hit = store.fts_search(middle.text.split()[0], limit=1)[0]

    expanded = store.expand_neighbors(
        [hit],
        neighbor_chunks=2,
        parent_section_max_tokens=middle.token_count,
    )

    assert [item.chunk_id for item in expanded] == [hit.chunk_id]
