from __future__ import annotations

from agent.config import SearchConfig
from agent.search import rerank as rerank_module
from agent.search.hybrid import HybridSearchService
from agent.search.rerank import RerankerBackendUnavailable
from agent.search.schema import SearchHit


class StubStore:
    def __init__(self, hits: list[SearchHit]):
        self.hits = hits

    def fts_search(self, query: str, *, limit: int) -> list[SearchHit]:
        return self.hits[:limit]


class StubVectorIndex:
    def __init__(self, hits: list[SearchHit]):
        self.hits = hits

    def search(self, query: str, embedding_provider, *, limit: int) -> list[SearchHit]:
        return self.hits[:limit]


class StubEmbeddingProvider:
    model = "stub"
    dimensions = 3
    provider_name = "stub"


def _hit(
    chunk_id: str,
    *,
    doc_id: str = "DOC-001",
    text: str = "candidate text",
    score: float = 0.1,
    source: str = "fts",
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        revision="A",
        title=f"{doc_id} title",
        section="Procedure",
        text=text,
        score=score,
        source=source,
        metadata={"filename": f"{doc_id}.docx"},
    )


def _service(
    lexical: list[SearchHit],
    vector: list[SearchHit],
    config: SearchConfig,
) -> HybridSearchService:
    return HybridSearchService(
        StubStore(lexical),
        StubVectorIndex(vector),
        StubEmbeddingProvider(),
        config,
    )


def test_search_with_trace_preserves_current_search_results():
    lexical = [
        _hit("chunk-a", text="sterility validation protocol", score=0.9),
        _hit("chunk-b", text="risk analysis summary", score=0.8),
    ]
    vector = [
        _hit("chunk-c", text="software verification report", score=0.7, source="faiss"),
        _hit("chunk-a", text="sterility validation protocol", score=0.6, source="faiss"),
    ]
    config = SearchConfig(
        reranker_enabled=False,
        reranker_top_n_candidates=4,
        reranker_top_k=4,
        hybrid_top_n_lexical=2,
        hybrid_top_n_vector=2,
    )
    service = _service(lexical, vector, config)

    direct = service.search("sterility protocol", limit=3)
    traced = service.search_with_trace("sterility protocol", limit=3)

    assert traced.hits == direct
    assert [hit.chunk_id for hit in direct] == ["chunk-a", "chunk-c", "chunk-b"]
    assert traced.trace.reranker_backend == "disabled"
    assert traced.trace.fallback_reason is None
    assert traced.trace.candidate_counts.lexical == 2
    assert traced.trace.candidate_counts.vector == 2
    assert traced.trace.candidate_counts.fused == 3
    assert traced.trace.candidate_counts.returned == 3
    assert traced.trace.lexical_candidates[0].pre_rank == 1
    assert traced.trace.lexical_candidates[0].pre_score == 0.9
    assert traced.trace.vector_candidates[0].pre_rank == 1
    assert traced.trace.fused_candidates[0].chunk_id == "chunk-a"
    assert traced.trace.fused_candidates[0].pre_rank == 1
    assert traced.trace.fused_candidates[0].pre_score == 0.9


def test_traced_search_records_fallback_backend_counts_and_candidates(monkeypatch):
    def unavailable(_config):
        raise RerankerBackendUnavailable("sentence_transformers is not installed")

    monkeypatch.setattr(rerank_module, "_load_cross_encoder_model", unavailable)
    lexical = [
        _hit("chunk-a", doc_id="DOC-001", text="unrelated text", score=0.9),
        _hit(
            "chunk-b",
            doc_id="BOM-055",
            text="sterility protocol and verification evidence",
            score=0.8,
        ),
    ]
    vector = [
        _hit(
            "chunk-c",
            doc_id="DOC-002",
            text="protocol appendix",
            score=0.7,
            source="faiss",
        )
    ]
    config = SearchConfig(
        reranker_top_n_candidates=3,
        reranker_top_k=2,
        hybrid_top_n_lexical=2,
        hybrid_top_n_vector=1,
    )
    service = _service(lexical, vector, config)

    traced = service.search_with_trace("Find BOM-055 sterility protocol", limit=2)

    assert [hit.chunk_id for hit in traced.hits] == ["chunk-b", "chunk-c"]
    assert traced.trace.reranker_backend == "deterministic_fallback"
    assert "sentence_transformers" in (traced.trace.fallback_reason or "")
    assert traced.trace.candidate_counts.lexical == 2
    assert traced.trace.candidate_counts.vector == 1
    assert traced.trace.candidate_counts.fused == 3
    assert traced.trace.candidate_counts.rerank_input == 3
    assert traced.trace.candidate_counts.rerank_limited == 3
    assert traced.trace.candidate_counts.reranked == 2
    assert traced.trace.candidate_counts.returned == 2
    assert [candidate.chunk_id for candidate in traced.trace.reranked_candidates] == [
        "chunk-b",
        "chunk-c",
    ]
    assert traced.trace.reranked_candidates[0].pre_rank == 3
    assert traced.trace.reranked_candidates[0].pre_score is not None
    assert traced.trace.reranked_candidates[0].rerank_score is not None
