from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from agent.config import SearchConfig
from agent.search.embeddings import EmbeddingProvider, normalize_vectors
from agent.search.index_contracts import (
    ARTIFACT_CONTRACT_VERSION,
    CHUNKING_CONTRACT_FIELDS,
    VECTOR_DATA_FILENAME,
    VECTOR_MANIFEST_ARTIFACT_TYPE,
    VECTOR_MANIFEST_FILENAME,
    VECTOR_MANIFEST_SCHEMA_VERSION,
    VECTOR_METADATA_FILENAME,
    validate_index_artifacts,
)
from agent.search.schema import Chunk, SearchHit


VECTOR_BACKEND_FAISS = "faiss"
VECTOR_BACKEND_NUMPY_FALLBACK = "numpy_fallback"


class LocalVectorIndex:
    def __init__(self, index_dir: Path, config: SearchConfig):
        self.index_dir = index_dir
        self.config = config
        self.index_path = index_dir / VECTOR_DATA_FILENAME
        self.metadata_path = index_dir / VECTOR_METADATA_FILENAME
        self.manifest_path = index_dir / VECTOR_MANIFEST_FILENAME

    def _load_faiss(self) -> Any | None:
        try:
            return importlib.import_module("faiss")
        except ImportError:
            return None

    def _current_vector_backend(self) -> str:
        if self._load_faiss() is not None:
            return VECTOR_BACKEND_FAISS
        return VECTOR_BACKEND_NUMPY_FALLBACK

    def _create_faiss_index(self, vectors: np.ndarray, faiss_module: Any) -> Any:
        index = faiss_module.IndexFlatIP(int(vectors.shape[1]))
        if vectors.shape[0] > 0:
            index.add(vectors)
        return index

    def _storage_description(self, vector_backend: str) -> str:
        if vector_backend == VECTOR_BACKEND_FAISS:
            return "vectors.npy + vector_metadata.json; search backend=faiss.IndexFlatIP"
        return (
            "vectors.npy + vector_metadata.json; "
            "search backend=numpy IndexFlatIP-compatible fallback"
        )

    def build(
        self,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
        *,
        corpus_hash: str = "",
    ) -> dict[str, object]:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vectors = embedding_provider.embed_texts(
            [chunk.search_text for chunk in chunks]
        )
        if vectors and len(vectors[0]) != self.config.embedding_dimensions:
            raise ValueError(
                "FAISS dimension mismatch: "
                f"expected {self.config.embedding_dimensions}, got {len(vectors[0])}"
            )
        normalized = (
            normalize_vectors(vectors)
            if vectors
            else np.empty((0, self.config.embedding_dimensions), dtype=np.float32)
        )
        normalized = np.ascontiguousarray(normalized, dtype=np.float32)
        faiss_module = self._load_faiss()
        vector_backend = (
            VECTOR_BACKEND_FAISS
            if faiss_module is not None
            else VECTOR_BACKEND_NUMPY_FALLBACK
        )
        if faiss_module is not None:
            self._create_faiss_index(normalized, faiss_module)
        np.save(self.index_path, normalized)
        metadata = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "revision": chunk.revision,
                "title": chunk.title,
                "section": chunk.section,
                "text": chunk.text,
                "search_text": chunk.search_text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        manifest = {
            "artifact_type": VECTOR_MANIFEST_ARTIFACT_TYPE,
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "schema_version": VECTOR_MANIFEST_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "embedding_provider": getattr(
                embedding_provider, "provider_name", "unknown"
            ),
            "embedding_model": self.config.embedding_model,
            "embedding_dimensions": self.config.embedding_dimensions,
            "embedding_batch_size": self.config.embedding_batch_size,
            "actual_embedding_dimensions": int(normalized.shape[1]),
            "vector_index": self.config.vector_index,
            "faiss_index_type": self.config.faiss_index_type,
            "vector_backend": vector_backend,
            "vectors_normalized": True,
            "chunk_count": len(chunks),
            "chunking": {
                "chunk_size_tokens": self.config.chunk_size_tokens,
                "chunk_overlap_tokens": self.config.chunk_overlap_tokens,
                "min_chunk_tokens": self.config.min_chunk_tokens,
                "max_chunk_tokens": self.config.max_chunk_tokens,
                "table_chunk_target_tokens": self.config.table_chunk_target_tokens,
                "table_chunk_max_tokens": self.config.table_chunk_max_tokens,
                "table_row_chunk_max_rows": self.config.table_row_chunk_max_rows,
                "table_repeat_header": self.config.table_repeat_header,
                "create_metadata_chunks": self.config.create_metadata_chunks,
                "parent_section_max_tokens": self.config.parent_section_max_tokens,
                "answer_context_neighbor_chunks": (
                    self.config.answer_context_neighbor_chunks
                ),
            },
            "corpus_hash": corpus_hash,
            "storage": self._storage_description(vector_backend),
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    def exists(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    def manifest(self) -> dict[str, object] | None:
        if not self.manifest_path.exists():
            return None
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def search(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        *,
        limit: int = 20,
    ) -> list[SearchHit]:
        if not self.exists():
            return []
        vectors = np.ascontiguousarray(np.load(self.index_path), dtype=np.float32)
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if vectors.size == 0 or limit <= 0:
            return []
        query_vector = embedding_provider.embed_query(query)
        if len(query_vector) != vectors.shape[1]:
            raise ValueError(
                "Query embedding dimension mismatch: "
                f"expected {vectors.shape[1]}, got {len(query_vector)}"
            )
        query_matrix = np.ascontiguousarray(
            normalize_vectors([query_vector]), dtype=np.float32
        )
        faiss_module = self._load_faiss()
        if faiss_module is not None:
            faiss_index = self._create_faiss_index(vectors, faiss_module)
            k = min(limit, len(metadata), int(vectors.shape[0]))
            if k <= 0:
                return []
            score_matrix, index_matrix = faiss_index.search(query_matrix, k)
            scores = score_matrix[0]
            indexes = index_matrix[0]
        else:
            scores = vectors @ query_matrix[0]
            indexes = np.argsort(scores)[::-1][:limit]
        hits: list[SearchHit] = []
        for position, index in enumerate(indexes):
            if int(index) < 0 or int(index) >= len(metadata):
                continue
            item = metadata[int(index)]
            item_metadata = item.get("metadata", {})
            hits.append(
                SearchHit(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    revision=item["revision"],
                    title=item["title"],
                    section=item["section"],
                    text=item["text"],
                    score=float(
                        scores[position]
                        if faiss_module is not None
                        else scores[int(index)]
                    ),
                    source="faiss",
                    metadata=item_metadata,
                    evidence_type=_evidence_type(item, item_metadata),
                    support_level=_support_level(item, item_metadata),
                    table_index=_optional_int(
                        item.get("table_index") or item_metadata.get("table_index")
                    ),
                    row_start=_optional_int(
                        item.get("row_start") or item_metadata.get("row_start")
                    ),
                    row_end=_optional_int(
                        item.get("row_end") or item_metadata.get("row_end")
                    ),
                    heading_path=tuple(
                        str(value)
                        for value in (
                            item.get("heading_path")
                            or item_metadata.get("heading_path")
                            or []
                        )
                        if value
                    ),
                    columns=tuple(
                        str(value)
                        for value in (
                            item.get("columns") or item_metadata.get("columns") or []
                        )
                        if value
                    ),
                    row_cells={
                        str(key): str(value)
                        for key, value in (
                            item.get("row_cells")
                            or item_metadata.get("row_cells")
                            or {}
                        ).items()
                    },
                )
            )
        return hits

    def stats(self) -> dict[str, object]:
        manifest = self.manifest()
        validation = self.validation()
        if manifest is None:
            return {"exists": False, "validation": validation}
        manifest = dict(manifest)
        active_vector_backend = self._current_vector_backend()
        vector_backend = str(
            manifest.get("vector_backend") or active_vector_backend
        )
        manifest.setdefault("vector_backend", vector_backend)
        manifest["active_vector_backend"] = active_vector_backend
        if not manifest.get("storage") or str(manifest["storage"]).startswith(
            "numpy-compatible"
        ):
            manifest["storage"] = self._storage_description(vector_backend)
        return {"exists": True, **manifest, "validation": validation}

    def validation(self) -> dict[str, object]:
        return validate_index_artifacts(
            self.index_dir,
            expected_corpus_zip=self.config.corpus_zip,
            expected_embedding_dimensions=self.config.embedding_dimensions,
            expected_embedding_model=self.config.embedding_model,
            expected_vector_index=self.config.vector_index,
            expected_faiss_index_type=self.config.faiss_index_type,
            expected_chunking={
                field: getattr(self.config, field) for field in CHUNKING_CONTRACT_FIELDS
            },
        )

    def validate_production_ready(self) -> None:
        manifest = self.manifest()
        if manifest is None:
            raise RuntimeError("production search requires a built local vector index")
        provider = manifest.get("embedding_provider")
        if provider != "openai":
            raise RuntimeError(
                "production search requires an OpenAI embedding index; "
                f"current embedding_provider={provider!r}"
            )
        actual_dimensions = manifest.get("actual_embedding_dimensions")
        if actual_dimensions != self.config.embedding_dimensions:
            raise RuntimeError(
                "production search index dimension mismatch: "
                f"config={self.config.embedding_dimensions}, manifest={actual_dimensions}"
            )


def _evidence_type(item: dict[str, Any], metadata: dict[str, Any]) -> str:
    value = item.get("evidence_type") or metadata.get("evidence_type")
    if isinstance(value, str) and value:
        return value
    kind = str(item.get("kind") or metadata.get("kind") or "")
    if kind == "metadata":
        return "metadata"
    if kind == "table_row":
        return "table_row"
    if kind == "table":
        return "table_full"
    return "prose"


def _support_level(item: dict[str, Any], metadata: dict[str, Any]) -> str:
    value = item.get("support_level") or metadata.get("support_level")
    if isinstance(value, str) and value:
        return value
    kind = str(item.get("kind") or metadata.get("kind") or "")
    if kind == "metadata":
        return "document"
    if kind == "table_row":
        return "row"
    if kind == "table":
        return "table"
    return "chunk"


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
