from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agent.search.schema import SearchHit


@dataclass(frozen=True)
class RetrievalTraceCandidate:
    chunk_id: str
    doc_id: str
    revision: str
    title: str
    section: str
    source: str
    rank: int
    score: float
    pre_rank: int | None = None
    pre_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class RetrievalCandidateCounts:
    lexical: int = 0
    vector: int = 0
    fused: int = 0
    rerank_input: int = 0
    rerank_limited: int = 0
    reranked: int = 0
    returned: int = 0


@dataclass(frozen=True)
class RetrievalTrace:
    lexical_candidates: list[RetrievalTraceCandidate] = field(default_factory=list)
    vector_candidates: list[RetrievalTraceCandidate] = field(default_factory=list)
    fused_candidates: list[RetrievalTraceCandidate] = field(default_factory=list)
    reranked_candidates: list[RetrievalTraceCandidate] = field(default_factory=list)
    reranker_backend: str = "unknown"
    fallback_reason: str | None = None
    candidate_counts: RetrievalCandidateCounts = field(
        default_factory=RetrievalCandidateCounts
    )


@dataclass(frozen=True)
class TracedSearchResult:
    hits: list[SearchHit]
    trace: RetrievalTrace


@dataclass(frozen=True)
class RerankTraceResult:
    hits: list[SearchHit]
    candidates: list[RetrievalTraceCandidate]
    backend: str
    fallback_reason: str | None
    input_candidate_count: int
    limited_candidate_count: int


def candidates_from_hits(
    hits: Sequence[SearchHit],
    *,
    pre_ranks: Mapping[str, int] | None = None,
    pre_scores: Mapping[str, float] | None = None,
    rerank_scores: Mapping[str, float] | None = None,
) -> list[RetrievalTraceCandidate]:
    return [
        RetrievalTraceCandidate(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            revision=hit.revision,
            title=hit.title,
            section=hit.section,
            source=hit.source,
            rank=rank,
            score=float(hit.score),
            pre_rank=pre_ranks.get(hit.chunk_id) if pre_ranks else None,
            pre_score=(
                float(pre_scores[hit.chunk_id])
                if pre_scores and hit.chunk_id in pre_scores
                else None
            ),
            rerank_score=(
                float(rerank_scores[hit.chunk_id])
                if rerank_scores and hit.chunk_id in rerank_scores
                else None
            ),
        )
        for rank, hit in enumerate(hits, start=1)
    ]
