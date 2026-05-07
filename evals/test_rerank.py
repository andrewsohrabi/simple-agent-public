from __future__ import annotations

from agent.config import SearchConfig
from agent.search import rerank as rerank_module
from agent.search.rerank import LocalReranker, RerankerBackendUnavailable
from agent.search.schema import SearchHit


def _hit(
    chunk_id: str,
    *,
    doc_id: str = "DOC-001",
    text: str = "candidate text",
    score: float = 0.1,
    source: str = "hybrid",
    metadata: dict[str, object] | None = None,
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
        metadata=metadata or {"filename": f"{doc_id}.docx"},
    )


def test_real_cross_encoder_backend_reranks_limited_candidates(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self):
            self.pairs = None

        def predict(self, pairs):
            self.pairs = pairs
            return [0.2, 0.95, 0.4]

    fake_model = FakeCrossEncoder()
    monkeypatch.setattr(
        rerank_module, "_load_cross_encoder_model", lambda _config: fake_model
    )
    hits = [
        _hit("chunk-a", text="risk analysis summary"),
        _hit("chunk-b", text="sterility validation protocol"),
        _hit("chunk-c", text="software verification report"),
        _hit("chunk-d", text="excluded candidate"),
    ]
    config = SearchConfig(reranker_top_n_candidates=3, reranker_top_k=2)

    reranker = LocalReranker(config)
    results = reranker.rerank("sterility protocol", hits)

    assert [hit.chunk_id for hit in results] == ["chunk-b", "chunk-c"]
    assert fake_model.pairs == [
        ("sterility protocol", "risk analysis summary"),
        ("sterility protocol", "sterility validation protocol"),
        ("sterility protocol", "software verification report"),
    ]
    assert reranker.status()["backend"] == "sentence_transformers_cross_encoder"
    assert reranker.status().get("warning") is None


def test_real_cross_encoder_trace_records_scores_and_pre_rank(monkeypatch):
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.2, 0.95, 0.4]

    monkeypatch.setattr(
        rerank_module, "_load_cross_encoder_model", lambda _config: FakeCrossEncoder()
    )
    hits = [
        _hit("chunk-a", text="risk analysis summary", score=0.3),
        _hit("chunk-b", text="sterility validation protocol", score=0.2),
        _hit("chunk-c", text="software verification report", score=0.1),
        _hit("chunk-d", text="excluded candidate", score=0.9),
    ]
    reranker = LocalReranker(
        SearchConfig(reranker_top_n_candidates=3, reranker_top_k=2)
    )

    result = reranker.rerank_with_trace("sterility protocol", hits)

    assert [hit.chunk_id for hit in result.hits] == ["chunk-b", "chunk-c"]
    assert result.backend == "sentence_transformers_cross_encoder"
    assert result.fallback_reason is None
    assert result.input_candidate_count == 4
    assert result.limited_candidate_count == 3
    assert [candidate.chunk_id for candidate in result.candidates] == [
        "chunk-b",
        "chunk-c",
    ]
    assert result.candidates[0].pre_rank == 2
    assert result.candidates[0].pre_score == 0.2
    assert result.candidates[0].rerank_score == 0.95


def test_disabled_reranker_preserves_input_order_without_loading_backend(monkeypatch):
    def fail_if_loaded(_config):
        raise AssertionError("disabled reranker should not load a backend")

    monkeypatch.setattr(rerank_module, "_load_cross_encoder_model", fail_if_loaded)
    hits = [_hit("chunk-a"), _hit("chunk-b"), _hit("chunk-c")]
    config = SearchConfig(
        reranker_enabled=False,
        reranker_top_n_candidates=3,
        reranker_top_k=2,
    )

    reranker = LocalReranker(config)
    results = reranker.rerank("any query", hits)

    assert [hit.chunk_id for hit in results] == ["chunk-a", "chunk-b"]
    assert reranker.status()["enabled"] is False
    assert reranker.status()["backend"] == "disabled"


def test_dependency_unavailable_falls_back_with_status_reason(monkeypatch):
    def unavailable(_config):
        raise RerankerBackendUnavailable("sentence_transformers is not installed")

    monkeypatch.setattr(rerank_module, "_load_cross_encoder_model", unavailable)
    hits = [
        _hit("chunk-a", doc_id="DOC-001", text="unrelated text", score=0.1),
        _hit(
            "chunk-b",
            doc_id="BOM-055",
            text="sterility protocol and verification evidence",
            score=0.1,
        ),
        _hit("chunk-c", doc_id="DOC-002", text="protocol appendix", score=0.1),
    ]

    reranker = LocalReranker(
        SearchConfig(reranker_top_n_candidates=3, reranker_top_k=2)
    )
    results = reranker.rerank("Find BOM-055 sterility protocol", hits)
    status = reranker.status()

    assert [hit.chunk_id for hit in results] == ["chunk-b", "chunk-c"]
    assert status["backend"] == "deterministic_fallback"
    assert status["configured_model"] == "Qwen/Qwen3-Reranker-4B"
    assert status["warning"] == "real_reranker_unavailable"
    assert "sentence_transformers" in status["fallback_reason"]


def test_deterministic_fallback_trace_exposes_reason_and_scores(monkeypatch):
    def unavailable(_config):
        raise RerankerBackendUnavailable("sentence_transformers is not installed")

    monkeypatch.setattr(rerank_module, "_load_cross_encoder_model", unavailable)
    hits = [
        _hit("chunk-a", doc_id="DOC-001", text="unrelated text", score=0.1),
        _hit(
            "chunk-b",
            doc_id="BOM-055",
            text="sterility protocol and verification evidence",
            score=0.1,
        ),
        _hit("chunk-c", doc_id="DOC-002", text="protocol appendix", score=0.1),
    ]
    reranker = LocalReranker(
        SearchConfig(reranker_top_n_candidates=3, reranker_top_k=2)
    )

    result = reranker.rerank_with_trace("Find BOM-055 sterility protocol", hits)

    assert [hit.chunk_id for hit in result.hits] == ["chunk-b", "chunk-c"]
    assert result.backend == "deterministic_fallback"
    assert "sentence_transformers" in (result.fallback_reason or "")
    assert result.input_candidate_count == 3
    assert result.limited_candidate_count == 3
    assert result.candidates[0].chunk_id == "chunk-b"
    assert result.candidates[0].pre_rank == 2
    assert result.candidates[0].pre_score == 0.1
    assert result.candidates[0].rerank_score is not None


def test_rerank_preserves_hit_identity_source_metadata_and_limits(monkeypatch):
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.1, 0.9]

    monkeypatch.setattr(
        rerank_module, "_load_cross_encoder_model", lambda _config: FakeCrossEncoder()
    )
    metadata = {"filename": "BOM-055.docx", "chunk_index": 7}
    hits = [
        _hit("chunk-a", metadata={"filename": "DOC-001.docx"}),
        _hit("chunk-b", doc_id="BOM-055", source="vector", metadata=metadata),
        _hit("chunk-c", metadata={"filename": "excluded.docx"}),
    ]

    reranker = LocalReranker(
        SearchConfig(reranker_top_n_candidates=2, reranker_top_k=1)
    )
    results = reranker.rerank("BOM sterility protocol", hits)

    assert results == [hits[1]]
    assert results[0].chunk_id == "chunk-b"
    assert results[0].source == "vector"
    assert results[0].metadata is metadata
