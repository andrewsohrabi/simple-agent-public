import sys
import types

import numpy as np

from agent.config import SearchConfig
from agent.search.embeddings import HashEmbeddingProvider
from agent.search.faiss_store import LocalVectorIndex
from agent.search.schema import Chunk


class OpenAIFakeEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self, dimensions: int, model: str):
        super().__init__(dimensions, model=model)
        self.provider_name = "openai"


class StaticEmbeddingProvider:
    provider_name = "static"
    model = "static-test"

    def __init__(
        self,
        text_vectors: dict[str, list[float]],
        query_vectors: dict[str, list[float]],
    ):
        self.text_vectors = text_vectors
        self.query_vectors = query_vectors
        first_vector = next(iter({**text_vectors, **query_vectors}.values()))
        self.dimensions = len(first_vector)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.text_vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.query_vectors[text]


def chunk(chunk_id: str, doc_id: str, search_text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        revision="A",
        title=f"{doc_id} title",
        text=search_text,
        search_text=search_text,
        section="Document",
        ordinal=0,
        metadata={},
    )


def fake_faiss_module():
    instances = []

    class FakeIndexFlatIP:
        def __init__(self, dimensions: int):
            self.dimensions = dimensions
            self.added = None
            self.search_calls = []
            instances.append(self)

        def add(self, vectors):
            self.added = np.asarray(vectors, dtype=np.float32).copy()

        def search(self, queries, limit: int):
            query_array = np.asarray(queries, dtype=np.float32)
            self.search_calls.append((query_array.copy(), limit))
            scores = query_array @ self.added.T
            indexes = np.argsort(scores, axis=1)[:, ::-1][:, :limit]
            sorted_scores = np.take_along_axis(scores, indexes, axis=1).astype(
                np.float32
            )
            return sorted_scores, indexes.astype(np.int64)

    return types.SimpleNamespace(IndexFlatIP=FakeIndexFlatIP, instances=instances)


def test_local_vector_index_manifest_records_model_and_dimensions(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", None)
    config = SearchConfig(index_dir=tmp_path)
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="BOM-055",
            revision="G",
            title="MX1 Top-level assembly",
            text="Document ID: BOM-055 Bill of Materials MX1 system",
            search_text="Document ID: BOM-055 Bill of Materials MX1 system",
            section="Document",
            ordinal=0,
            metadata={"filename": "BOM-055.docx"},
        )
    ]
    index = LocalVectorIndex(tmp_path, config)
    manifest = index.build(
        chunks,
        HashEmbeddingProvider(
            config.embedding_dimensions, model=config.embedding_model
        ),
    )
    assert manifest["embedding_provider"] == "hash"
    assert manifest["embedding_model"] == "text-embedding-3-large"
    assert manifest["embedding_dimensions"] == 3072
    assert manifest["actual_embedding_dimensions"] == 3072
    assert manifest["faiss_index_type"] == "IndexFlatIP"
    assert manifest["vectors_normalized"] is True
    assert manifest["vector_backend"] == "numpy_fallback"
    hits = index.search(
        "Find BOM-055",
        HashEmbeddingProvider(
            config.embedding_dimensions, model=config.embedding_model
        ),
    )
    assert hits[0].doc_id == "BOM-055"


def test_local_vector_index_uses_faiss_backend_when_importable(tmp_path, monkeypatch):
    fake_faiss = fake_faiss_module()
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    config = SearchConfig(index_dir=tmp_path, embedding_dimensions=2)
    chunks = [
        chunk("c1", "DOC-A", "alpha"),
        chunk("c2", "DOC-B", "beta"),
    ]
    provider = StaticEmbeddingProvider(
        {"alpha": [3.0, 0.0], "beta": [0.0, 4.0]},
        {"needle": [2.0, 0.0]},
    )

    manifest = LocalVectorIndex(tmp_path, config).build(chunks, provider)

    assert manifest["vector_backend"] == "faiss"
    assert "faiss.IndexFlatIP" in manifest["storage"]
    assert (tmp_path / "vectors.npy").exists()
    assert (tmp_path / "vector_metadata.json").exists()
    assert fake_faiss.instances[0].dimensions == 2
    np.testing.assert_allclose(
        fake_faiss.instances[0].added, [[1.0, 0.0], [0.0, 1.0]]
    )
    assert fake_faiss.instances[0].added.dtype == np.float32

    hits = LocalVectorIndex(tmp_path, config).search("needle", provider)
    stats = LocalVectorIndex(tmp_path, config).stats()

    assert [hit.doc_id for hit in hits] == ["DOC-A", "DOC-B"]
    assert fake_faiss.instances[-1].search_calls
    assert stats["vector_backend"] == "faiss"


def test_local_vector_index_uses_numpy_fallback_when_faiss_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", None)
    config = SearchConfig(index_dir=tmp_path, embedding_dimensions=2)
    chunks = [
        chunk("c1", "DOC-A", "alpha"),
        chunk("c2", "DOC-B", "beta"),
    ]
    provider = StaticEmbeddingProvider(
        {"alpha": [3.0, 0.0], "beta": [0.0, 4.0]},
        {"needle": [0.0, 2.0]},
    )

    manifest = LocalVectorIndex(tmp_path, config).build(chunks, provider)
    hits = LocalVectorIndex(tmp_path, config).search("needle", provider)
    stats = LocalVectorIndex(tmp_path, config).stats()

    assert manifest["vector_backend"] == "numpy_fallback"
    assert "numpy" in manifest["storage"]
    assert stats["vector_backend"] == "numpy_fallback"
    assert [hit.doc_id for hit in hits] == ["DOC-B", "DOC-A"]


def test_local_vector_index_rejects_dimension_mismatch(tmp_path):
    config = SearchConfig(index_dir=tmp_path, embedding_dimensions=3072)
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="BOM-055",
            revision="G",
            title="MX1 Top-level assembly",
            text="Bill of Materials",
            search_text="Bill of Materials",
            section="Document",
            ordinal=0,
            metadata={},
        )
    ]
    index = LocalVectorIndex(tmp_path, config)
    try:
        index.build(
            chunks, HashEmbeddingProvider(16, model=config.embedding_model)
        )
    except ValueError as exc:
        assert "dimension mismatch" in str(exc).lower()
    else:
        raise AssertionError("expected dimension mismatch")


def test_production_validation_requires_openai_provider_and_dimensions(tmp_path):
    config = SearchConfig(index_dir=tmp_path, embedding_dimensions=32)
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="BOM-055",
            revision="G",
            title="MX1 Top-level assembly",
            text="Bill of Materials",
            search_text="Bill of Materials",
            section="Document",
            ordinal=0,
            metadata={},
        )
    ]
    index = LocalVectorIndex(tmp_path, config)
    index.build(chunks, HashEmbeddingProvider(32, model=config.embedding_model))

    try:
        index.validate_production_ready()
    except RuntimeError as exc:
        assert "OpenAI embedding index" in str(exc)
    else:
        raise AssertionError("expected production validation to reject hash index")

    index.build(
        chunks, OpenAIFakeEmbeddingProvider(32, model=config.embedding_model)
    )
    index.validate_production_ready()

    mismatched_config = SearchConfig(index_dir=tmp_path, embedding_dimensions=64)
    try:
        LocalVectorIndex(tmp_path, mismatched_config).validate_production_ready()
    except RuntimeError as exc:
        assert "dimension mismatch" in str(exc)
    else:
        raise AssertionError(
            "expected production validation to reject dimension mismatch"
        )
