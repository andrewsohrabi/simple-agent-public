from __future__ import annotations

from agent.config import SearchConfig
from agent.search.schema import SearchHit


class LocalReranker:
    """Lightweight deterministic reranker placeholder preserving the configured contract."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.backend = "deterministic_fallback"

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not self.config.reranker_enabled:
            return hits[: self.config.reranker_top_k]
        limited = hits[: self.config.reranker_top_n_candidates]
        query_terms = {term.strip(".,:;()[]").lower() for term in query.split() if term}

        def score(hit: SearchHit) -> float:
            text_terms = set(hit.text.lower().split())
            overlap = len(query_terms & text_terms)
            exact_id = 5 if hit.doc_id.lower() in query.lower() else 0
            return hit.score + overlap * 0.05 + exact_id

        return sorted(limited, key=score, reverse=True)[: self.config.reranker_top_k]

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.config.reranker_enabled,
            "configured_model": self.config.reranker_model,
            "backend": self.backend if self.config.reranker_enabled else "disabled",
            "top_n_candidates": self.config.reranker_top_n_candidates,
            "top_k": self.config.reranker_top_k,
            "max_length": self.config.reranker_max_length,
        }
