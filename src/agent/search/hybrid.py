from __future__ import annotations

from agent.config import SearchConfig
from agent.search.embeddings import EmbeddingProvider
from agent.search.faiss_store import LocalVectorIndex
from agent.search.rerank import LocalReranker
from agent.search.retrieval_trace import (
    RetrievalCandidateCounts,
    RetrievalTrace,
    TracedSearchResult,
    candidates_from_hits,
)
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

    def search(
        self,
        query: str,
        *,
        limit: int = 16,
        lexical_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> list[SearchHit]:
        _lexical, _vector, fused = self._candidate_sets(
            query,
            limit=limit,
            lexical_weight=lexical_weight,
            vector_weight=vector_weight,
        )
        return self.reranker.rerank(query, fused)[:limit]

    def search_with_trace(
        self,
        query: str,
        *,
        limit: int = 16,
        lexical_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> TracedSearchResult:
        lexical, vector, fused = self._candidate_sets(
            query,
            limit=limit,
            lexical_weight=lexical_weight,
            vector_weight=vector_weight,
        )
        reranked = self.reranker.rerank_with_trace(query, fused)
        hits = reranked.hits[:limit]
        fused_pre_ranks, fused_pre_scores = _fused_pre_maps(lexical, vector)

        return TracedSearchResult(
            hits=hits,
            trace=RetrievalTrace(
                lexical_candidates=candidates_from_hits(
                    lexical,
                    pre_ranks=_rank_map(lexical),
                    pre_scores=_score_map(lexical),
                ),
                vector_candidates=candidates_from_hits(
                    vector,
                    pre_ranks=_rank_map(vector),
                    pre_scores=_score_map(vector),
                ),
                fused_candidates=candidates_from_hits(
                    fused,
                    pre_ranks=fused_pre_ranks,
                    pre_scores=fused_pre_scores,
                ),
                reranked_candidates=reranked.candidates,
                reranker_backend=reranked.backend,
                fallback_reason=reranked.fallback_reason,
                candidate_counts=RetrievalCandidateCounts(
                    lexical=len(lexical),
                    vector=len(vector),
                    fused=len(fused),
                    rerank_input=reranked.input_candidate_count,
                    rerank_limited=reranked.limited_candidate_count,
                    reranked=len(reranked.hits),
                    returned=len(hits),
                ),
            ),
        )

    def _candidate_sets(
        self,
        query: str,
        *,
        limit: int,
        lexical_weight: float,
        vector_weight: float,
    ) -> tuple[list[SearchHit], list[SearchHit], list[SearchHit]]:
        lexical = self.store.fts_search(
            query, limit=max(limit, self.config.hybrid_top_n_lexical)
        )
        vector = self.vector_index.search(
            query,
            self.embedding_provider,
            limit=max(limit, self.config.hybrid_top_n_vector),
        )
        fused = self._fuse(
            lexical,
            vector,
            lexical_weight=lexical_weight,
            vector_weight=vector_weight,
        )
        return lexical, vector, fused

    def _fuse(
        self,
        lexical: list[SearchHit],
        vector: list[SearchHit],
        *,
        lexical_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> list[SearchHit]:
        by_chunk: dict[str, SearchHit] = {}
        scores: dict[str, float] = {}
        for rank, hit in enumerate(lexical, start=1):
            by_chunk[hit.chunk_id] = hit
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + lexical_weight / (60 + rank)
        for rank, hit in enumerate(vector, start=1):
            by_chunk.setdefault(hit.chunk_id, hit)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + vector_weight / (60 + rank)
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
                evidence_type=hit.evidence_type,
                support_level=hit.support_level,
                table_index=hit.table_index,
                row_start=hit.row_start,
                row_end=hit.row_end,
                heading_path=hit.heading_path,
                columns=hit.columns,
                row_cells=hit.row_cells,
            )
            for hit in ordered
        ]


def _rank_map(hits: list[SearchHit]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, hit in enumerate(hits, start=1):
        ranks.setdefault(hit.chunk_id, rank)
    return ranks


def _score_map(hits: list[SearchHit]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for hit in hits:
        scores.setdefault(hit.chunk_id, hit.score)
    return scores


def _fused_pre_maps(
    lexical: list[SearchHit],
    vector: list[SearchHit],
) -> tuple[dict[str, int], dict[str, float]]:
    ranks: dict[str, int] = {}
    scores: dict[str, float] = {}
    for candidates in (lexical, vector):
        for rank, hit in enumerate(candidates, start=1):
            ranks.setdefault(hit.chunk_id, rank)
            scores.setdefault(hit.chunk_id, hit.score)
    return ranks, scores
