from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


ARTIFACT_CONTRACT_VERSION = 1

INGEST_MANIFEST_FILENAME = "ingest_manifest.json"
INGEST_MANIFEST_ARTIFACT_TYPE = "qms_ingest_manifest"
INGEST_MANIFEST_SCHEMA_VERSION = 1

VECTOR_MANIFEST_FILENAME = "manifest.json"
VECTOR_MANIFEST_ARTIFACT_TYPE = "qms_vector_index_manifest"
VECTOR_MANIFEST_SCHEMA_VERSION = 1
VECTOR_DATA_FILENAME = "vectors.npy"
VECTOR_METADATA_FILENAME = "vector_metadata.json"

MANIFEST_VERSION_FIELDS = {
    "artifact_type",
    "artifact_contract_version",
    "schema_version",
}

INGEST_MANIFEST_REQUIRED_FIELDS = {
    *MANIFEST_VERSION_FIELDS,
    "created_at",
    "source_zip",
    "source_sha256",
    "document_count",
    "skipped_empty_count",
    "metadata_only_count",
    "normalized_dir",
    "documents",
}

VECTOR_MANIFEST_REQUIRED_FIELDS = {
    *MANIFEST_VERSION_FIELDS,
    "created_at",
    "embedding_provider",
    "embedding_model",
    "embedding_dimensions",
    "embedding_batch_size",
    "actual_embedding_dimensions",
    "vector_index",
    "faiss_index_type",
    "vector_backend",
    "vectors_normalized",
    "chunk_count",
    "chunking",
    "corpus_hash",
    "storage",
    "index_path",
    "metadata_path",
}

CHUNKING_CONTRACT_FIELDS = (
    "chunk_size_tokens",
    "chunk_overlap_tokens",
    "min_chunk_tokens",
    "max_chunk_tokens",
    "table_chunk_target_tokens",
    "table_chunk_max_tokens",
    "table_row_chunk_max_rows",
    "table_repeat_header",
    "create_metadata_chunks",
    "parent_section_max_tokens",
    "answer_context_neighbor_chunks",
)


def validate_index_artifacts(
    index_dir: Path,
    *,
    expected_corpus_zip: Path | None = None,
    expected_embedding_dimensions: int | None = None,
    expected_embedding_model: str | None = None,
    expected_vector_index: str | None = None,
    expected_faiss_index_type: str | None = None,
    expected_chunking: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Read index artifacts and report contract problems without mutating files."""
    index_dir = Path(index_dir)
    ingest_manifest_path = index_dir / INGEST_MANIFEST_FILENAME
    vector_manifest_path = index_dir / VECTOR_MANIFEST_FILENAME
    vectors_path = index_dir / VECTOR_DATA_FILENAME
    vector_metadata_path = index_dir / VECTOR_METADATA_FILENAME

    warnings: list[str] = []
    errors: list[str] = []
    checks: dict[str, object] = {
        "ingest_manifest": {"exists": ingest_manifest_path.exists()},
        "vector_manifest": {"exists": vector_manifest_path.exists()},
        "vectors": {"exists": vectors_path.exists()},
        "vector_metadata": {"exists": vector_metadata_path.exists()},
        "corpus_hash": {},
    }
    versions: dict[str, object] = {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "expected_ingest_schema_version": INGEST_MANIFEST_SCHEMA_VERSION,
        "expected_vector_schema_version": VECTOR_MANIFEST_SCHEMA_VERSION,
    }

    ingest_manifest = _read_json_object(
        ingest_manifest_path, "ingest_manifest", errors
    )
    vector_manifest = _read_json_object(
        vector_manifest_path, "vector_manifest", errors
    )

    if ingest_manifest is None and not ingest_manifest_path.exists():
        errors.append(f"missing ingest manifest: {ingest_manifest_path}")
    if vector_manifest is None and not vector_manifest_path.exists():
        errors.append(f"missing vector manifest: {vector_manifest_path}")

    if ingest_manifest is not None:
        _validate_manifest_contract(
            "ingest_manifest",
            ingest_manifest,
            required_fields=INGEST_MANIFEST_REQUIRED_FIELDS,
            expected_artifact_type=INGEST_MANIFEST_ARTIFACT_TYPE,
            expected_schema_version=INGEST_MANIFEST_SCHEMA_VERSION,
            errors=errors,
            warnings=warnings,
        )
        versions["ingest_schema_version"] = ingest_manifest.get("schema_version", 0)
        versions["ingest_artifact_contract_version"] = ingest_manifest.get(
            "artifact_contract_version", 0
        )
        checks["ingest_manifest"] = {
            **dict(checks["ingest_manifest"]),
            "document_count": ingest_manifest.get("document_count"),
            "source_sha256": ingest_manifest.get("source_sha256"),
        }

    if vector_manifest is not None:
        _validate_manifest_contract(
            "vector_manifest",
            vector_manifest,
            required_fields=VECTOR_MANIFEST_REQUIRED_FIELDS,
            expected_artifact_type=VECTOR_MANIFEST_ARTIFACT_TYPE,
            expected_schema_version=VECTOR_MANIFEST_SCHEMA_VERSION,
            errors=errors,
            warnings=warnings,
        )
        versions["vector_schema_version"] = vector_manifest.get("schema_version", 0)
        versions["vector_artifact_contract_version"] = vector_manifest.get(
            "artifact_contract_version", 0
        )
        checks["vector_manifest"] = {
            **dict(checks["vector_manifest"]),
            "chunk_count": vector_manifest.get("chunk_count"),
            "corpus_hash": vector_manifest.get("corpus_hash"),
        }
        _validate_expected_vector_config(
            vector_manifest,
            expected_embedding_dimensions=expected_embedding_dimensions,
            expected_embedding_model=expected_embedding_model,
            expected_vector_index=expected_vector_index,
            expected_faiss_index_type=expected_faiss_index_type,
            expected_chunking=expected_chunking,
            errors=errors,
        )

    metadata_count = _validate_vector_metadata(
        vector_metadata_path, vector_manifest, checks, errors
    )
    _validate_vectors(
        vectors_path,
        vector_manifest,
        metadata_count=metadata_count,
        expected_embedding_dimensions=expected_embedding_dimensions,
        checks=checks,
        errors=errors,
    )
    _validate_corpus_hashes(
        ingest_manifest,
        vector_manifest,
        expected_corpus_zip,
        checks,
        warnings,
        errors,
    )

    requires_rebuild = bool(errors)
    return {
        "ok": not errors,
        "requires_rebuild": requires_rebuild,
        "warnings": warnings,
        "errors": errors,
        "versions": versions,
        "artifacts": {
            "index_dir": str(index_dir),
            "ingest_manifest": str(ingest_manifest_path),
            "vector_manifest": str(vector_manifest_path),
            "vectors": str(vectors_path),
            "vector_metadata": str(vector_metadata_path),
        },
        "checks": checks,
    }


def _read_json_object(
    path: Path, artifact_name: str, errors: list[str]
) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{artifact_name} unreadable: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{artifact_name} must be a JSON object")
        return None
    return data


def _validate_manifest_contract(
    artifact_name: str,
    manifest: Mapping[str, object],
    *,
    required_fields: set[str],
    expected_artifact_type: str,
    expected_schema_version: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    for field in sorted(required_fields - set(manifest)):
        errors.append(f"{artifact_name} missing required field: {field}")

    artifact_type = manifest.get("artifact_type")
    if artifact_type is not None and artifact_type != expected_artifact_type:
        errors.append(
            f"{artifact_name} artifact_type mismatch: "
            f"expected {expected_artifact_type!r}, got {artifact_type!r}"
        )

    contract_version = _coerce_int(manifest.get("artifact_contract_version"))
    if contract_version is None:
        if "artifact_contract_version" in manifest:
            errors.append(f"{artifact_name} artifact_contract_version must be an integer")
    elif contract_version < ARTIFACT_CONTRACT_VERSION:
        errors.append(
            f"{artifact_name} stale artifact_contract_version: "
            f"expected {ARTIFACT_CONTRACT_VERSION}, got {contract_version}"
        )
    elif contract_version > ARTIFACT_CONTRACT_VERSION:
        warnings.append(
            f"{artifact_name} artifact_contract_version is newer than this code: "
            f"{contract_version}"
        )

    schema_version = _coerce_int(manifest.get("schema_version"))
    if schema_version is None:
        if "schema_version" in manifest:
            errors.append(f"{artifact_name} schema_version must be an integer")
        return
    if schema_version == 0:
        errors.append(f"{artifact_name} uses legacy schema version 0")
    elif schema_version < expected_schema_version:
        errors.append(
            f"{artifact_name} stale schema_version: "
            f"expected {expected_schema_version}, got {schema_version}"
        )
    elif schema_version > expected_schema_version:
        warnings.append(
            f"{artifact_name} schema_version is newer than this code: {schema_version}"
        )


def _validate_expected_vector_config(
    manifest: Mapping[str, object],
    *,
    expected_embedding_dimensions: int | None,
    expected_embedding_model: str | None,
    expected_vector_index: str | None,
    expected_faiss_index_type: str | None,
    expected_chunking: Mapping[str, object] | None,
    errors: list[str],
) -> None:
    _compare_expected_field(
        manifest,
        "embedding_dimensions",
        expected_embedding_dimensions,
        errors,
        "vector_manifest stale field",
    )
    _compare_expected_field(
        manifest,
        "actual_embedding_dimensions",
        expected_embedding_dimensions,
        errors,
        "vector_manifest stale field",
    )
    _compare_expected_field(
        manifest,
        "embedding_model",
        expected_embedding_model,
        errors,
        "vector_manifest stale field",
    )
    _compare_expected_field(
        manifest,
        "vector_index",
        expected_vector_index,
        errors,
        "vector_manifest stale field",
    )
    _compare_expected_field(
        manifest,
        "faiss_index_type",
        expected_faiss_index_type,
        errors,
        "vector_manifest stale field",
    )
    if expected_chunking is None:
        return
    chunking = manifest.get("chunking")
    if not isinstance(chunking, Mapping):
        errors.append("vector_manifest chunking must be an object")
        return
    for field, expected in expected_chunking.items():
        _compare_expected_field(
            chunking,
            field,
            expected,
            errors,
            "vector_manifest stale chunking field",
        )


def _validate_vector_metadata(
    path: Path,
    vector_manifest: Mapping[str, object] | None,
    checks: dict[str, object],
    errors: list[str],
) -> int | None:
    if not path.exists():
        errors.append(f"missing vector metadata: {path}")
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"vector metadata unreadable: {exc}")
        return None
    if not isinstance(metadata, list):
        errors.append("vector metadata must be a JSON list")
        return None
    metadata_count = len(metadata)
    checks["vector_metadata"] = {
        **dict(checks["vector_metadata"]),
        "count": metadata_count,
    }
    if vector_manifest is not None:
        chunk_count = _coerce_int(vector_manifest.get("chunk_count"))
        if chunk_count is not None and metadata_count != chunk_count:
            errors.append(
                "vector metadata length mismatch: "
                f"metadata={metadata_count}, manifest chunk_count={chunk_count}"
            )
    return metadata_count


def _validate_vectors(
    path: Path,
    vector_manifest: Mapping[str, object] | None,
    *,
    metadata_count: int | None,
    expected_embedding_dimensions: int | None,
    checks: dict[str, object],
    errors: list[str],
) -> None:
    if not path.exists():
        errors.append(f"missing vector data: {path}")
        return
    try:
        vectors = np.load(path, allow_pickle=False)
    except Exception as exc:
        errors.append(f"vector data unreadable: {exc}")
        return
    shape = tuple(int(value) for value in vectors.shape)
    checks["vectors"] = {
        **dict(checks["vectors"]),
        "shape": list(shape),
        "dtype": str(vectors.dtype),
    }
    if vectors.ndim != 2:
        errors.append(f"vector shape mismatch: expected 2-D array, got shape {shape}")
        return
    row_count, dimension_count = shape
    if metadata_count is not None and row_count != metadata_count:
        errors.append(
            "vector shape mismatch: "
            f"rows={row_count}, vector_metadata={metadata_count}"
        )
    if vector_manifest is not None:
        chunk_count = _coerce_int(vector_manifest.get("chunk_count"))
        if chunk_count is not None and row_count != chunk_count:
            errors.append(
                "vector shape mismatch: "
                f"rows={row_count}, manifest chunk_count={chunk_count}"
            )
        actual_dimensions = _coerce_int(
            vector_manifest.get("actual_embedding_dimensions")
        )
        if actual_dimensions is not None and dimension_count != actual_dimensions:
            errors.append(
                "vector shape mismatch: "
                f"dimensions={dimension_count}, "
                f"manifest actual_embedding_dimensions={actual_dimensions}"
            )
        embedding_dimensions = _coerce_int(vector_manifest.get("embedding_dimensions"))
        if embedding_dimensions is not None and dimension_count != embedding_dimensions:
            errors.append(
                "vector shape mismatch: "
                f"dimensions={dimension_count}, "
                f"manifest embedding_dimensions={embedding_dimensions}"
            )
    if (
        expected_embedding_dimensions is not None
        and dimension_count != expected_embedding_dimensions
    ):
        errors.append(
            "vector shape mismatch: "
            f"dimensions={dimension_count}, config={expected_embedding_dimensions}"
        )


def _validate_corpus_hashes(
    ingest_manifest: Mapping[str, object] | None,
    vector_manifest: Mapping[str, object] | None,
    expected_corpus_zip: Path | None,
    checks: dict[str, object],
    warnings: list[str],
    errors: list[str],
) -> None:
    corpus_check = dict(checks["corpus_hash"])
    ingest_hash = _string_or_none(
        ingest_manifest.get("source_sha256") if ingest_manifest else None
    )
    vector_hash = _string_or_none(
        vector_manifest.get("corpus_hash") if vector_manifest else None
    )
    if ingest_hash:
        corpus_check["ingest_source_sha256"] = ingest_hash
    if vector_hash:
        corpus_check["vector_corpus_hash"] = vector_hash
    if ingest_hash and vector_hash and ingest_hash != vector_hash:
        errors.append(
            "corpus hash mismatch: "
            f"ingest source_sha256={ingest_hash}, vector corpus_hash={vector_hash}"
        )

    if expected_corpus_zip is not None:
        expected_corpus_zip = Path(expected_corpus_zip)
        corpus_check["configured_corpus_zip"] = str(expected_corpus_zip)
        corpus_check["configured_corpus_zip_exists"] = expected_corpus_zip.exists()
        if not expected_corpus_zip.exists():
            warnings.append(f"configured corpus zip is missing: {expected_corpus_zip}")
        elif ingest_hash or vector_hash:
            current_hash = hashlib.sha256(expected_corpus_zip.read_bytes()).hexdigest()
            corpus_check["configured_corpus_sha256"] = current_hash
            if ingest_hash and ingest_hash != current_hash:
                errors.append(
                    "corpus hash mismatch: "
                    "ingest source_sha256 does not match configured corpus zip"
                )
            if vector_hash and vector_hash != current_hash:
                errors.append(
                    "corpus hash mismatch: "
                    "vector corpus_hash does not match configured corpus zip"
                )
    checks["corpus_hash"] = corpus_check


def _compare_expected_field(
    manifest: Mapping[str, object],
    field: str,
    expected: object | None,
    errors: list[str],
    prefix: str,
) -> None:
    if expected is None or field not in manifest:
        return
    actual = manifest.get(field)
    if actual != expected:
        errors.append(f"{prefix}: {field} expected {expected!r}, got {actual!r}")


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
