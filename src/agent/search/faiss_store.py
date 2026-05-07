from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from agent.config import SearchConfig
from agent.search.embeddings import EmbeddingProvider, normalize_vectors
from agent.search.schema import Chunk, SearchHit


class LocalVectorIndex:
    def __init__(self, index_dir: Path, config: SearchConfig):
        self.index_dir = index_dir
        self.config = config
        self.index_path = index_dir / "vectors.npy"
        self.metadata_path = index_dir / "vector_metadata.json"
        self.manifest_path = index_dir / "manifest.json"

    def build(
        self,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider,
        *,
        corpus_hash: str = "",
    ) -> dict[str, object]:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vectors = embedding_provider.embed_texts([chunk.search_text for chunk in chunks])
        if vectors and len(vectors[0]) != self.config.embedding_dimensions:
            raise ValueError(
                "FAISS dimension mismatch: "
                f"expected {self.config.embedding_dimensions}, got {len(vectors[0])}"
            )
        normalized = normalize_vectors(vectors) if vectors else np.empty((0, self.config.embedding_dimensions), dtype=np.float32)
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
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "embedding_model": self.config.embedding_model,
            "embedding_dimensions": self.config.embedding_dimensions,
            "actual_embedding_dimensions": int(normalized.shape[1]),
            "vector_index": self.config.vector_index,
            "faiss_index_type": self.config.faiss_index_type,
            "vectors_normalized": True,
            "chunk_count": len(chunks),
            "chunking": {
                "chunk_size_tokens": self.config.chunk_size_tokens,
                "chunk_overlap_tokens": self.config.chunk_overlap_tokens,
                "min_chunk_tokens": self.config.min_chunk_tokens,
                "max_chunk_tokens": self.config.max_chunk_tokens,
                "table_chunk_target_tokens": self.config.table_chunk_target_tokens,
                "table_chunk_max_tokens": self.config.table_chunk_max_tokens,
                "table_repeat_header": self.config.table_repeat_header,
                "create_metadata_chunks": self.config.create_metadata_chunks,
                "parent_section_max_tokens": self.config.parent_section_max_tokens,
                "answer_context_neighbor_chunks": self.config.answer_context_neighbor_chunks,
            },
            "corpus_hash": corpus_hash,
            "storage": "numpy-compatible IndexFlatIP",
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def exists(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    def manifest(self) -> dict[str, object] | None:
        if not self.manifest_path.exists():
            return None
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def search(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        *,
        limit: int = 20,
    ) -> list[SearchHit]:
        if not self.exists():
            return []
        vectors = np.load(self.index_path)
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if vectors.size == 0:
            return []
        query_vector = embedding_provider.embed_query(query)
        if len(query_vector) != vectors.shape[1]:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {vectors.shape[1]}, got {len(query_vector)}"
            )
        query_matrix = normalize_vectors([query_vector])
        scores = vectors @ query_matrix[0]
        indexes = np.argsort(scores)[::-1][:limit]
        hits: list[SearchHit] = []
        for index in indexes:
            item = metadata[int(index)]
            hits.append(
                SearchHit(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    revision=item["revision"],
                    title=item["title"],
                    section=item["section"],
                    text=item["text"],
                    score=float(scores[int(index)]),
                    source="faiss",
                    metadata=item.get("metadata", {}),
                )
            )
        return hits

    def stats(self) -> dict[str, object]:
        manifest = self.manifest()
        if manifest is None:
            return {"exists": False}
        return {"exists": True, **manifest}
