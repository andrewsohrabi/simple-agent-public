import json
import sys
import types

import pytest

from agent.config import SearchConfig
from agent.search.hosted_sync import sync_openai_vector_store
from agent.search.openai_file_search import OpenAIFileSearch, metadata_from_markdown
from agent.search.sqlite_store import SearchStore


def test_metadata_from_markdown_parses_source_mapping(tmp_path):
    markdown = tmp_path / "BOM-055_rev-G.md"
    markdown.write_text(
        "\n".join(
            [
                "# BOM-055 Rev G: MX1 Top-level assembly",
                "",
                "## Metadata",
                "- Document ID: BOM-055",
                "- Revision: G",
                "- Prefix: BOM",
                "- Latest revision: True",
                "- Signed: False",
                "- Obsolete: False",
                "- Source filename: BOM-055 - MX1 Top-level assembly_G.docx",
                "- Source path: Example QMS - MedAI/BOM-055 - MX1 Top-level assembly_G.docx",
            ]
        ),
        encoding="utf-8",
    )

    metadata = metadata_from_markdown(markdown)

    assert metadata["doc_id"] == "BOM-055"
    assert metadata["revision"] == "G"
    assert metadata["prefix"] == "BOM"
    assert metadata["is_latest"] is True
    assert metadata["is_obsolete"] is False
    assert metadata["filename"] == "BOM-055 - MX1 Top-level assembly_G.docx"


def test_hosted_sync_reuses_matching_state_without_openai_client(tmp_path):
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir()
    markdown = normalized_dir / "BOM-055_rev-G.md"
    markdown.write_text("# BOM-055 Rev G\n", encoding="utf-8")
    state_path = tmp_path / "openai" / "vector_store_state.json"
    state_path.parent.mkdir()
    state = {
        "status": "synced",
        "vector_store_id": "vs_existing",
        "corpus_hash": "abc123",
        "file_count": 1,
        "file_signatures": [
            {
                "path": str(markdown),
                "name": markdown.name,
                "sha256": "45e7291bd2b5a904ee0a4a6eb706a5e9b8796a53e1e54e27d860d276e9cbfd8d",
                "bytes": markdown.stat().st_size,
            }
        ],
        "files": [{"file_id": "file_1", "doc_id": "BOM-055", "revision": "G"}],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    config = SearchConfig(
        index_dir=tmp_path,
        openai_vector_store_state=state_path,
    )

    result = sync_openai_vector_store(config, normalized_dir, corpus_hash="abc123")

    assert result == state


def test_hosted_sync_recovers_from_invalid_state_json(tmp_path, monkeypatch):
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir()
    state_path = tmp_path / "openai" / "vector_store_state.json"
    state_path.parent.mkdir()
    state_path.write_text("{", encoding="utf-8")

    class VectorStores:
        def create(self, *, name):
            return types.SimpleNamespace(id="vs_recovered")

    class FakeOpenAI:
        def __init__(self):
            self.vector_stores = VectorStores()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    config = SearchConfig(
        index_dir=tmp_path,
        openai_vector_store_state=state_path,
    )

    state = sync_openai_vector_store(config, normalized_dir, corpus_hash="hash")

    assert state["status"] == "synced"
    assert state["vector_store_id"] == "vs_recovered"
    assert state["file_count"] == 0


def test_hosted_sync_uploads_only_normalized_markdown_with_mock_openai(tmp_path, monkeypatch):
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir()
    markdown = normalized_dir / "BOM-055_rev-G.md"
    markdown.write_text(
        "\n".join(
            [
                "# BOM-055 Rev G: MX1 Top-level assembly",
                "- Document ID: BOM-055",
                "- Revision: G",
                "- Prefix: BOM",
                "- Source filename: BOM-055_G.docx",
                "- Source path: Example QMS/BOM-055_G.docx",
            ]
        ),
        encoding="utf-8",
    )
    (normalized_dir / "raw.docx").write_bytes(b"not uploaded")
    calls = {"uploads": [], "attributes": []}

    class Files:
        def create(self, *, file, purpose):
            calls["uploads"].append((file.name, purpose, file.read()))
            return types.SimpleNamespace(id=f"file_{len(calls['uploads'])}")

    class VectorStoreFiles:
        def create_and_poll(self, **kwargs):
            calls["attributes"].append(kwargs)

    class VectorStores:
        files = VectorStoreFiles()

        def create(self, *, name):
            return types.SimpleNamespace(id="vs_new")

    class FakeOpenAI:
        def __init__(self):
            self.files = Files()
            self.vector_stores = VectorStores()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    config = SearchConfig(
        index_dir=tmp_path,
        openai_vector_store_state=tmp_path / "openai" / "vector_store_state.json",
    )

    state = sync_openai_vector_store(config, normalized_dir, corpus_hash="hash")

    assert state["status"] == "synced"
    assert state["source_format"] == "normalized_markdown"
    assert state["vector_store_id"] == "vs_new"
    assert state["file_count"] == 1
    assert calls["uploads"] == [(str(markdown), "assistants", markdown.read_bytes())]
    assert calls["attributes"][0]["attributes"]["doc_id"] == "BOM-055"
    assert calls["attributes"][0]["attributes"]["source_path"] == "Example QMS/BOM-055_G.docx"


def test_hosted_sync_cleans_up_previous_hosted_assets_after_success(tmp_path, monkeypatch):
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir()
    markdown = normalized_dir / "BOM-055_rev-G.md"
    markdown.write_text(
        "\n".join(
            [
                "# BOM-055 Rev G: MX1 Top-level assembly",
                "- Document ID: BOM-055",
                "- Revision: G",
                "- Prefix: BOM",
                "- Source filename: BOM-055_G.docx",
                "- Source path: Example QMS/BOM-055_G.docx",
            ]
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "openai" / "vector_store_state.json"
    state_path.parent.mkdir()
    state_path.write_text(
        json.dumps(
            {
                "status": "synced",
                "vector_store_id": "vs_old",
                "corpus_hash": "old",
                "file_count": 1,
                "file_signatures": [],
                "files": [{"file_id": "file_old", "doc_id": "BOM-055"}],
            }
        ),
        encoding="utf-8",
    )
    calls = {"deleted_vector_stores": [], "deleted_files": []}

    class Files:
        def create(self, *, file, purpose):
            return types.SimpleNamespace(id="file_new")

        def delete(self, file_id):
            calls["deleted_files"].append(file_id)

    class VectorStoreFiles:
        def create_and_poll(self, **kwargs):
            return None

    class VectorStores:
        files = VectorStoreFiles()

        def create(self, *, name):
            return types.SimpleNamespace(id="vs_new")

        def delete(self, vector_store_id):
            calls["deleted_vector_stores"].append(vector_store_id)

    class FakeOpenAI:
        def __init__(self):
            self.files = Files()
            self.vector_stores = VectorStores()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    config = SearchConfig(
        index_dir=tmp_path,
        openai_vector_store_state=state_path,
    )

    state = sync_openai_vector_store(config, normalized_dir, corpus_hash="new")

    assert state["status"] == "synced"
    assert state["vector_store_id"] == "vs_new"
    assert calls["deleted_vector_stores"] == ["vs_old"]
    assert calls["deleted_files"] == ["file_old"]
    assert state["previous_cleanup"]["deleted_vector_store_id"] == "vs_old"
    assert state["previous_cleanup"]["deleted_file_ids"] == ["file_old"]


def test_hosted_sync_cleans_up_new_assets_after_sync_failure(tmp_path, monkeypatch):
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir()
    markdown = normalized_dir / "BOM-055_rev-G.md"
    markdown.write_text(
        "\n".join(
            [
                "# BOM-055 Rev G: MX1 Top-level assembly",
                "- Document ID: BOM-055",
                "- Revision: G",
                "- Prefix: BOM",
            ]
        ),
        encoding="utf-8",
    )
    calls = {"deleted_vector_stores": [], "deleted_files": []}

    class Files:
        def create(self, *, file, purpose):
            return types.SimpleNamespace(id="file_new")

        def delete(self, file_id):
            calls["deleted_files"].append(file_id)

    class VectorStoreFiles:
        def create_and_poll(self, **kwargs):
            raise RuntimeError("upload failed")

    class VectorStores:
        files = VectorStoreFiles()

        def create(self, *, name):
            return types.SimpleNamespace(id="vs_new")

        def delete(self, vector_store_id):
            calls["deleted_vector_stores"].append(vector_store_id)

    class FakeOpenAI:
        def __init__(self):
            self.files = Files()
            self.vector_stores = VectorStores()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    state_path = tmp_path / "openai" / "vector_store_state.json"
    config = SearchConfig(
        index_dir=tmp_path,
        openai_vector_store_state=state_path,
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        sync_openai_vector_store(config, normalized_dir, corpus_hash="hash")

    assert calls["deleted_vector_stores"] == ["vs_new"]
    assert calls["deleted_files"] == ["file_new"]
    failed_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert failed_state["status"] == "failed"
    assert failed_state["failure_cleanup"]["deleted_vector_store_id"] == "vs_new"
    assert failed_state["failure_cleanup"]["deleted_file_ids"] == ["file_new"]


def test_openai_file_search_maps_hosted_result_to_local_citation_metadata(
    tmp_path, monkeypatch
):
    store = SearchStore(tmp_path / "qms.sqlite")
    store.initialize()
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
                "MX1 Top-level assembly",
                7,
                "BOM-055",
                1,
                0,
                0,
                "BOM-055_G.docx",
                "Example QMS/BOM-055_G.docx",
                None,
                ".data/qms-index/normalized/BOM-055_rev-G.md",
                "sha",
            ),
        )
        conn.execute(
            """
            INSERT INTO chunks
            (chunk_id, doc_id, revision, title, section, ordinal, text, search_text,
             parent_section_id, kind, token_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "chunk-1",
                "BOM-055",
                "G",
                "MX1 Top-level assembly",
                "Metadata",
                0,
                "Filename: BOM-055_G.docx",
                "BOM-055 BOM-055_G.docx",
                "parent",
                "metadata",
                12,
                json.dumps({"filename": "BOM-055_G.docx"}),
            ),
        )
    state_path = tmp_path / "openai" / "vector_store_state.json"
    state_path.parent.mkdir()
    state_path.write_text(
        json.dumps(
            {
                "status": "synced",
                "vector_store_id": "vs_123",
                "files": [
                    {
                        "file_id": "file_1",
                        "doc_id": "BOM-055",
                        "revision": "G",
                        "filename": "BOM-055_G.docx",
                        "path": ".data/qms-index/normalized/BOM-055_rev-G.md",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class SearchResults:
        data = [
            {
                "file_id": "file_1",
                "filename": "BOM-055_rev-G.md",
                "score": 0.87,
                "content": [
                    {
                        "type": "text",
                        "text": "BOM-055 result",
                        "annotations": [{"type": "file_citation", "file_id": "file_1"}],
                    }
                ],
            }
        ]

    class VectorStores:
        def search(self, vector_store_id, **kwargs):
            assert vector_store_id == "vs_123"
            assert kwargs["query"] == "top assembly"
            assert kwargs["max_num_results"] == 3
            return SearchResults()

    class FakeOpenAI:
        def __init__(self):
            self.vector_stores = VectorStores()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    config = SearchConfig(index_dir=tmp_path, openai_vector_store_state=state_path)

    hits = OpenAIFileSearch(config).search("top assembly", store, limit=3)

    assert len(hits) == 1
    assert hits[0].chunk_id == "chunk-1"
    assert hits[0].source == "hosted_file_search"
    assert hits[0].score == 0.87
    assert hits[0].metadata["hosted_vector_store_id"] == "vs_123"
    assert hits[0].metadata["hosted_file_id"] == "file_1"
    assert hits[0].metadata["hosted_annotations"] == [
        {"type": "file_citation", "file_id": "file_1"}
    ]


def test_openai_file_search_clamps_hosted_result_limit(tmp_path, monkeypatch):
    store = SearchStore(tmp_path / "qms.sqlite")
    store.initialize()
    state_path = tmp_path / "openai" / "vector_store_state.json"
    state_path.parent.mkdir()
    state_path.write_text(
        json.dumps(
            {
                "status": "synced",
                "vector_store_id": "vs_123",
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    max_result_calls = []

    class SearchResults:
        data = []

    class VectorStores:
        def search(self, vector_store_id, **kwargs):
            assert vector_store_id == "vs_123"
            max_result_calls.append(kwargs["max_num_results"])
            return SearchResults()

    class FakeOpenAI:
        def __init__(self):
            self.vector_stores = VectorStores()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    config = SearchConfig(index_dir=tmp_path, openai_vector_store_state=state_path)

    OpenAIFileSearch(config).search("top assembly", store, limit=0)
    OpenAIFileSearch(config).search("top assembly", store, limit=99)

    assert max_result_calls == [1, 50]
