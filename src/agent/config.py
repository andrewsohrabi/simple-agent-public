from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True)
class SearchConfig:
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    chat_model: str = "gpt-5.5"
    agent_model: str = "gpt-5.5"
    enrichment_model: str = "gpt-5.5"
    eval_grader_model: str = "gpt-5.5"
    query_model: str = "gpt-5.4-mini"
    vector_index: str = "faiss"
    faiss_index_type: str = "IndexFlatIP"
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 100
    min_chunk_tokens: int = 120
    max_chunk_tokens: int = 900
    table_chunk_target_tokens: int = 700
    table_chunk_max_tokens: int = 900
    table_repeat_header: bool = True
    create_metadata_chunks: bool = True
    parent_section_max_tokens: int = 1800
    answer_context_neighbor_chunks: int = 1
    hybrid_top_n_lexical: int = 80
    hybrid_top_n_vector: int = 80
    hybrid_top_n_metadata: int = 30
    answer_max_chunks: int = 12
    reranker_enabled: bool = True
    reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    reranker_top_n_candidates: int = 80
    reranker_top_k: int = 16
    reranker_max_length: int = 4096
    corpus_zip: Path = Path("Example_QMS_-_MedAI.zip")
    index_dir: Path = Path(".data/qms-index")
    openai_vector_store_state: Path = Path(".data/openai/vector_store_state.json")

    @classmethod
    def from_env(cls) -> "SearchConfig":
        config = cls(
            embedding_model=_env("EMBEDDING_MODEL", cls.embedding_model),
            embedding_dimensions=_env_int(
                "EMBEDDING_DIMENSIONS", cls.embedding_dimensions
            ),
            chat_model=_env("CHAT_MODEL", cls.chat_model),
            agent_model=_env("AGENT_MODEL", cls.agent_model),
            enrichment_model=_env("ENRICHMENT_MODEL", cls.enrichment_model),
            eval_grader_model=_env("EVAL_GRADER_MODEL", cls.eval_grader_model),
            query_model=_env("QUERY_MODEL", cls.query_model),
            vector_index=_env("VECTOR_INDEX", cls.vector_index),
            faiss_index_type=_env("FAISS_INDEX_TYPE", cls.faiss_index_type),
            chunk_size_tokens=_env_int("CHUNK_SIZE_TOKENS", cls.chunk_size_tokens),
            chunk_overlap_tokens=_env_int(
                "CHUNK_OVERLAP_TOKENS", cls.chunk_overlap_tokens
            ),
            min_chunk_tokens=_env_int("MIN_CHUNK_TOKENS", cls.min_chunk_tokens),
            max_chunk_tokens=_env_int("MAX_CHUNK_TOKENS", cls.max_chunk_tokens),
            table_chunk_target_tokens=_env_int(
                "TABLE_CHUNK_TARGET_TOKENS", cls.table_chunk_target_tokens
            ),
            table_chunk_max_tokens=_env_int(
                "TABLE_CHUNK_MAX_TOKENS", cls.table_chunk_max_tokens
            ),
            table_repeat_header=_env_bool(
                "TABLE_REPEAT_HEADER", cls.table_repeat_header
            ),
            create_metadata_chunks=_env_bool(
                "CREATE_METADATA_CHUNKS", cls.create_metadata_chunks
            ),
            parent_section_max_tokens=_env_int(
                "PARENT_SECTION_MAX_TOKENS", cls.parent_section_max_tokens
            ),
            answer_context_neighbor_chunks=_env_int(
                "ANSWER_CONTEXT_NEIGHBOR_CHUNKS",
                cls.answer_context_neighbor_chunks,
            ),
            hybrid_top_n_lexical=_env_int(
                "HYBRID_TOP_N_LEXICAL", cls.hybrid_top_n_lexical
            ),
            hybrid_top_n_vector=_env_int(
                "HYBRID_TOP_N_VECTOR", cls.hybrid_top_n_vector
            ),
            hybrid_top_n_metadata=_env_int(
                "HYBRID_TOP_N_METADATA", cls.hybrid_top_n_metadata
            ),
            answer_max_chunks=_env_int("ANSWER_MAX_CHUNKS", cls.answer_max_chunks),
            reranker_enabled=_env_bool("RERANKER_ENABLED", cls.reranker_enabled),
            reranker_model=_env("RERANKER_MODEL", cls.reranker_model),
            reranker_top_n_candidates=_env_int(
                "RERANKER_TOP_N_CANDIDATES", cls.reranker_top_n_candidates
            ),
            reranker_top_k=_env_int("RERANKER_TOP_K", cls.reranker_top_k),
            reranker_max_length=_env_int(
                "RERANKER_MAX_LENGTH", cls.reranker_max_length
            ),
            corpus_zip=Path(_env("QMS_CORPUS_ZIP", str(cls.corpus_zip))),
            index_dir=Path(_env("QMS_INDEX_DIR", str(cls.index_dir))),
            openai_vector_store_state=Path(
                _env(
                    "OPENAI_VECTOR_STORE_STATE",
                    str(cls.openai_vector_store_state),
                )
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.embedding_dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be positive")
        if self.vector_index != "faiss":
            raise ValueError("VECTOR_INDEX must be 'faiss' for this implementation")
        if self.faiss_index_type != "IndexFlatIP":
            raise ValueError("FAISS_INDEX_TYPE must be 'IndexFlatIP'")
        positive_fields = {
            "CHUNK_SIZE_TOKENS": self.chunk_size_tokens,
            "MIN_CHUNK_TOKENS": self.min_chunk_tokens,
            "MAX_CHUNK_TOKENS": self.max_chunk_tokens,
            "TABLE_CHUNK_TARGET_TOKENS": self.table_chunk_target_tokens,
            "TABLE_CHUNK_MAX_TOKENS": self.table_chunk_max_tokens,
            "PARENT_SECTION_MAX_TOKENS": self.parent_section_max_tokens,
            "HYBRID_TOP_N_LEXICAL": self.hybrid_top_n_lexical,
            "HYBRID_TOP_N_VECTOR": self.hybrid_top_n_vector,
            "HYBRID_TOP_N_METADATA": self.hybrid_top_n_metadata,
            "ANSWER_MAX_CHUNKS": self.answer_max_chunks,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be non-negative")
        if self.answer_context_neighbor_chunks < 0:
            raise ValueError("ANSWER_CONTEXT_NEIGHBOR_CHUNKS must be non-negative")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.min_chunk_tokens > self.max_chunk_tokens:
            raise ValueError("MIN_CHUNK_TOKENS cannot exceed MAX_CHUNK_TOKENS")
        if self.chunk_size_tokens > self.max_chunk_tokens:
            raise ValueError("CHUNK_SIZE_TOKENS cannot exceed MAX_CHUNK_TOKENS")
        if self.table_chunk_target_tokens > self.table_chunk_max_tokens:
            raise ValueError(
                "TABLE_CHUNK_TARGET_TOKENS cannot exceed TABLE_CHUNK_MAX_TOKENS"
            )
        if self.reranker_top_n_candidates <= 0:
            raise ValueError("RERANKER_TOP_N_CANDIDATES must be positive")
        if self.reranker_top_k <= 0:
            raise ValueError("RERANKER_TOP_K must be positive")
        if self.reranker_top_k > self.reranker_top_n_candidates:
            raise ValueError("RERANKER_TOP_K cannot exceed RERANKER_TOP_N_CANDIDATES")
        if self.reranker_max_length <= 0:
            raise ValueError("RERANKER_MAX_LENGTH must be positive")

    def model_config(self) -> dict[str, object]:
        return {
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "vector_index": self.vector_index,
            "faiss_index_type": self.faiss_index_type,
            "chunk_size_tokens": self.chunk_size_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "min_chunk_tokens": self.min_chunk_tokens,
            "max_chunk_tokens": self.max_chunk_tokens,
            "table_chunk_target_tokens": self.table_chunk_target_tokens,
            "table_chunk_max_tokens": self.table_chunk_max_tokens,
            "table_repeat_header": self.table_repeat_header,
            "create_metadata_chunks": self.create_metadata_chunks,
            "parent_section_max_tokens": self.parent_section_max_tokens,
            "answer_context_neighbor_chunks": self.answer_context_neighbor_chunks,
            "hybrid_top_n_lexical": self.hybrid_top_n_lexical,
            "hybrid_top_n_vector": self.hybrid_top_n_vector,
            "hybrid_top_n_metadata": self.hybrid_top_n_metadata,
            "answer_max_chunks": self.answer_max_chunks,
            "chat_model": self.chat_model,
            "agent_model": self.agent_model,
            "enrichment_model": self.enrichment_model,
            "eval_grader_model": self.eval_grader_model,
            "query_model": self.query_model,
            "reranker_enabled": self.reranker_enabled,
            "reranker_model": self.reranker_model,
            "reranker_top_n_candidates": self.reranker_top_n_candidates,
            "reranker_top_k": self.reranker_top_k,
            "reranker_max_length": self.reranker_max_length,
        }

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["corpus_zip"] = str(self.corpus_zip)
        data["index_dir"] = str(self.index_dir)
        data["openai_vector_store_state"] = str(self.openai_vector_store_state)
        return data


def load_config() -> SearchConfig:
    return SearchConfig.from_env()
