from agent.config import SearchConfig
from agent.search.embeddings import HashEmbeddingProvider
from agent.search.faiss_store import LocalVectorIndex
from agent.search.schema import Chunk


def test_local_vector_index_manifest_records_model_and_dimensions(tmp_path):
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
    manifest = index.build(chunks, HashEmbeddingProvider(config.embedding_dimensions, model=config.embedding_model))
    assert manifest["embedding_model"] == "text-embedding-3-large"
    assert manifest["embedding_dimensions"] == 3072
    assert manifest["actual_embedding_dimensions"] == 3072
    assert manifest["faiss_index_type"] == "IndexFlatIP"
    assert manifest["vectors_normalized"] is True
    hits = index.search(
        "Find BOM-055",
        HashEmbeddingProvider(config.embedding_dimensions, model=config.embedding_model),
    )
    assert hits[0].doc_id == "BOM-055"


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
        index.build(chunks, HashEmbeddingProvider(16, model=config.embedding_model))
    except ValueError as exc:
        assert "dimension mismatch" in str(exc).lower()
    else:
        raise AssertionError("expected dimension mismatch")
