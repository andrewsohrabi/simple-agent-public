from agent.config import SearchConfig


def test_config_defaults_to_quality_baseline():
    config = SearchConfig()
    assert config.embedding_model == "text-embedding-3-large"
    assert config.embedding_dimensions == 3072
    assert config.embedding_batch_size == 128
    assert config.chat_model == "gpt-5.5"
    assert config.agent_model == "gpt-5.5"
    assert config.enrichment_model == "gpt-5.5"
    assert config.eval_grader_model == "gpt-5.5"
    assert config.query_model == "gpt-5.4-mini"
    assert config.vector_index == "faiss"
    assert config.faiss_index_type == "IndexFlatIP"
    assert config.chunk_size_tokens == 600
    assert config.chunk_overlap_tokens == 100
    assert config.min_chunk_tokens == 120
    assert config.max_chunk_tokens == 900
    assert config.table_chunk_target_tokens == 700
    assert config.table_chunk_max_tokens == 900
    assert config.create_metadata_chunks is True
    assert config.answer_context_neighbor_chunks == 1
    assert config.reranker_enabled is True
    assert config.reranker_model == "Qwen/Qwen3-Reranker-4B"
    assert config.use_hash_embeddings is False
    assert config.runtime_env == "development"


def test_config_validates_reranker_top_k():
    config = SearchConfig(reranker_top_n_candidates=10, reranker_top_k=11)
    try:
        config.validate()
    except ValueError as exc:
        assert "RERANKER_TOP_K" in str(exc)
    else:
        raise AssertionError("expected invalid reranker config to fail")


def test_production_config_rejects_hash_embeddings():
    config = SearchConfig(runtime_env="production", use_hash_embeddings=True)
    try:
        config.validate()
    except ValueError as exc:
        assert "QMS_USE_HASH_EMBEDDINGS" in str(exc)
    else:
        raise AssertionError("expected production hash config to fail")
