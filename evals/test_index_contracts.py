import hashlib
import json

import numpy as np

from agent.config import SearchConfig
from agent.search.index_contracts import (
    ARTIFACT_CONTRACT_VERSION,
    CHUNKING_CONTRACT_FIELDS,
    INGEST_MANIFEST_ARTIFACT_TYPE,
    INGEST_MANIFEST_FILENAME,
    INGEST_MANIFEST_SCHEMA_VERSION,
    VECTOR_DATA_FILENAME,
    VECTOR_MANIFEST_ARTIFACT_TYPE,
    VECTOR_MANIFEST_FILENAME,
    VECTOR_MANIFEST_SCHEMA_VERSION,
    VECTOR_METADATA_FILENAME,
    validate_index_artifacts,
)
from agent.search.stats import collect_stats


def _chunking(config: SearchConfig) -> dict[str, object]:
    return {field: getattr(config, field) for field in CHUNKING_CONTRACT_FIELDS}


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_artifacts(
    tmp_path,
    config: SearchConfig,
    *,
    chunk_count: int = 2,
    dimensions: int = 3,
    metadata_count: int | None = None,
    vector_rows: int | None = None,
    vector_dimensions: int | None = None,
    ingest_schema_version: int = INGEST_MANIFEST_SCHEMA_VERSION,
    vector_schema_version: int = VECTOR_MANIFEST_SCHEMA_VERSION,
    vector_corpus_hash: str | None = None,
) -> str:
    corpus_hash = hashlib.sha256(config.corpus_zip.read_bytes()).hexdigest()
    vector_corpus_hash = (
        vector_corpus_hash if vector_corpus_hash is not None else corpus_hash
    )
    ingest_manifest = {
        "artifact_type": INGEST_MANIFEST_ARTIFACT_TYPE,
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "schema_version": ingest_schema_version,
        "created_at": "2026-05-07T00:00:00+00:00",
        "source_zip": str(config.corpus_zip),
        "source_sha256": corpus_hash,
        "document_count": 1,
        "skipped_empty_count": 0,
        "metadata_only_count": 0,
        "metadata_only": [],
        "normalized_dir": str(tmp_path / "normalized"),
        "documents": [],
    }
    vector_manifest = {
        "artifact_type": VECTOR_MANIFEST_ARTIFACT_TYPE,
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "schema_version": vector_schema_version,
        "created_at": "2026-05-07T00:00:00+00:00",
        "embedding_provider": "hash",
        "embedding_model": config.embedding_model,
        "embedding_dimensions": dimensions,
        "embedding_batch_size": config.embedding_batch_size,
        "actual_embedding_dimensions": dimensions,
        "vector_index": config.vector_index,
        "faiss_index_type": config.faiss_index_type,
        "vector_backend": "numpy_fallback",
        "vectors_normalized": True,
        "chunk_count": chunk_count,
        "chunking": _chunking(config),
        "corpus_hash": vector_corpus_hash,
        "storage": "vectors.npy + vector_metadata.json",
        "index_path": str(tmp_path / VECTOR_DATA_FILENAME),
        "metadata_path": str(tmp_path / VECTOR_METADATA_FILENAME),
    }
    _write_json(tmp_path / INGEST_MANIFEST_FILENAME, ingest_manifest)
    _write_json(tmp_path / VECTOR_MANIFEST_FILENAME, vector_manifest)
    metadata_count = chunk_count if metadata_count is None else metadata_count
    _write_json(
        tmp_path / VECTOR_METADATA_FILENAME,
        [{"chunk_id": f"c{index}"} for index in range(metadata_count)],
    )
    vector_rows = chunk_count if vector_rows is None else vector_rows
    vector_dimensions = dimensions if vector_dimensions is None else vector_dimensions
    np.save(
        tmp_path / VECTOR_DATA_FILENAME,
        np.zeros((vector_rows, vector_dimensions), dtype=np.float32),
    )
    return corpus_hash


def test_validate_index_artifacts_accepts_current_contract(tmp_path):
    corpus_zip = tmp_path / "corpus.zip"
    corpus_zip.write_bytes(b"current corpus")
    config = SearchConfig(
        index_dir=tmp_path,
        corpus_zip=corpus_zip,
        embedding_dimensions=3,
    )
    _write_artifacts(tmp_path, config, dimensions=3)

    report = validate_index_artifacts(
        tmp_path,
        expected_corpus_zip=corpus_zip,
        expected_embedding_dimensions=config.embedding_dimensions,
        expected_embedding_model=config.embedding_model,
        expected_vector_index=config.vector_index,
        expected_faiss_index_type=config.faiss_index_type,
        expected_chunking=_chunking(config),
    )

    assert report["ok"] is True
    assert report["requires_rebuild"] is False
    assert report["warnings"] == []
    assert report["errors"] == []
    assert report["checks"]["vectors"]["shape"] == [2, 3]


def test_validate_index_artifacts_reports_legacy_and_missing_schema_fields(tmp_path):
    corpus_zip = tmp_path / "corpus.zip"
    corpus_zip.write_bytes(b"current corpus")
    config = SearchConfig(
        index_dir=tmp_path,
        corpus_zip=corpus_zip,
        embedding_dimensions=3,
    )
    _write_artifacts(
        tmp_path,
        config,
        dimensions=3,
        ingest_schema_version=0,
        vector_schema_version=0,
    )
    vector_manifest_path = tmp_path / VECTOR_MANIFEST_FILENAME
    vector_manifest = json.loads(vector_manifest_path.read_text(encoding="utf-8"))
    del vector_manifest["embedding_model"]
    _write_json(vector_manifest_path, vector_manifest)

    report = validate_index_artifacts(
        tmp_path,
        expected_corpus_zip=corpus_zip,
        expected_embedding_dimensions=config.embedding_dimensions,
        expected_embedding_model=config.embedding_model,
        expected_vector_index=config.vector_index,
        expected_faiss_index_type=config.faiss_index_type,
        expected_chunking=_chunking(config),
    )

    assert report["ok"] is False
    assert report["requires_rebuild"] is True
    assert any("legacy schema version 0" in error for error in report["errors"])
    assert any(
        "vector_manifest missing required field: embedding_model" in error
        for error in report["errors"]
    )


def test_validate_index_artifacts_reports_vector_length_and_shape_mismatches(
    tmp_path,
):
    corpus_zip = tmp_path / "corpus.zip"
    corpus_zip.write_bytes(b"current corpus")
    config = SearchConfig(
        index_dir=tmp_path,
        corpus_zip=corpus_zip,
        embedding_dimensions=3,
    )
    _write_artifacts(
        tmp_path,
        config,
        chunk_count=2,
        dimensions=3,
        metadata_count=1,
        vector_rows=3,
        vector_dimensions=4,
    )

    report = validate_index_artifacts(
        tmp_path,
        expected_corpus_zip=corpus_zip,
        expected_embedding_dimensions=config.embedding_dimensions,
        expected_embedding_model=config.embedding_model,
        expected_vector_index=config.vector_index,
        expected_faiss_index_type=config.faiss_index_type,
        expected_chunking=_chunking(config),
    )

    assert report["requires_rebuild"] is True
    assert any("vector metadata length mismatch" in error for error in report["errors"])
    assert any("vector shape mismatch: rows=3" in error for error in report["errors"])
    assert any(
        "vector shape mismatch: dimensions=4" in error for error in report["errors"]
    )


def test_validate_index_artifacts_reports_corpus_hash_mismatch(tmp_path):
    corpus_zip = tmp_path / "corpus.zip"
    corpus_zip.write_bytes(b"current corpus")
    config = SearchConfig(
        index_dir=tmp_path,
        corpus_zip=corpus_zip,
        embedding_dimensions=3,
    )
    _write_artifacts(
        tmp_path,
        config,
        dimensions=3,
        vector_corpus_hash="stale-vector-hash",
    )

    report = validate_index_artifacts(
        tmp_path,
        expected_corpus_zip=corpus_zip,
        expected_embedding_dimensions=config.embedding_dimensions,
        expected_embedding_model=config.embedding_model,
        expected_vector_index=config.vector_index,
        expected_faiss_index_type=config.faiss_index_type,
        expected_chunking=_chunking(config),
    )

    assert report["ok"] is False
    assert report["requires_rebuild"] is True
    assert any("corpus hash mismatch" in error for error in report["errors"])


def test_collect_stats_exposes_artifact_validation(tmp_path):
    corpus_zip = tmp_path / "corpus.zip"
    corpus_zip.write_bytes(b"current corpus")
    config = SearchConfig(
        index_dir=tmp_path,
        corpus_zip=corpus_zip,
        embedding_dimensions=3,
        openai_vector_store_state=tmp_path / "openai-state.json",
        reranker_enabled=False,
    )
    _write_artifacts(tmp_path, config, dimensions=3)

    stats = collect_stats(config)

    assert stats["artifact_validation"]["ok"] is True
    assert stats["vector_index"]["validation"]["ok"] is True
