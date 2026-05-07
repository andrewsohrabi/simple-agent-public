from __future__ import annotations

from agent.config import SearchConfig
from agent.search.embeddings import EmbeddingProvider
from agent.search.faiss_store import LocalVectorIndex
from agent.search.rerank import LocalReranker
from agent.search.schema import SearchHit
from agent.search.sqlite_store import SearchStore


class HybridSearchService:
    def __init__(
        self,
        store: SearchStore,
        vector_index: LocalVectorIndex,
        embedding_provider: EmbeddingProvider,
        config: SearchConfig,
    ):
        self.store = store
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider
        self.config = config
        self.reranker = LocalReranker(config)

    def search(self, query: str, *, limit: int = 16) -> list[SearchHit]:
        lexical = self.store.fts_search(query, limit=max(limit, self.config.hybrid_top_n_lexical))
        vector = self.vector_index.search(
            query, self.embedding_provider, limit=max(limit, self.config.hybrid_top_n_vector)
        )
        fused = self._fuse(lexical, vector)
        return self.reranker.rerank(query, fused)[:limit]

    def _fuse(self, lexical: list[SearchHit], vector: list[SearchHit]) -> list[SearchHit]:
        by_chunk: dict[str, SearchHit] = {}
        scores: dict[str, float] = {}
        for rank, hit in enumerate(lexical, start=1):
            by_chunk[hit.chunk_id] = hit
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (60 + rank)
        for rank, hit in enumerate(vector, start=1):
            by_chunk.setdefault(hit.chunk_id, hit)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (60 + rank)
        ordered = sorted(by_chunk.values(), key=lambda hit: scores[hit.chunk_id], reverse=True)
        return [
            SearchHit(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                revision=hit.revision,
                title=hit.title,
                section=hit.section,
                text=hit.text,
                score=scores[hit.chunk_id],
                source="hybrid",
                metadata=hit.metadata,
            )
            for hit in ordered
        ]
