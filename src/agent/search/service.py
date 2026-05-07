from __future__ import annotations

from agent.config import SearchConfig
from agent.search.answer import SearchAnswerer
from agent.search.embeddings import HashEmbeddingProvider, OpenAIEmbeddingProvider
from agent.search.faiss_store import LocalVectorIndex
from agent.search.hybrid import HybridSearchService
from agent.search.query_plan import plan_query
from agent.search.schema import SearchHit
from agent.search.sqlite_store import SearchStore


class QmsSearchService:
    def __init__(self, config: SearchConfig, *, use_hash_embeddings: bool = False):
        self.config = config
        self.store = SearchStore(config.index_dir / "qms.sqlite")
        self.vector_index = LocalVectorIndex(config.index_dir, config)
        self.embedding_provider = (
            HashEmbeddingProvider(config.embedding_dimensions, model=config.embedding_model)
            if use_hash_embeddings
            else OpenAIEmbeddingProvider(config)
        )
        self.hybrid = HybridSearchService(
            self.store, self.vector_index, self.embedding_provider, config
        )
        self.answerer = SearchAnswerer(self.store)

    def search(self, query: str, *, mode: str = "auto", limit: int = 16) -> dict[str, object]:
        plan = plan_query(query)
        hits: list[SearchHit] = []
        warnings: list[str] = []

        if plan.strategy == "sql_count" and plan.prefix:
            docs = self.store.count_by_prefix(plan.prefix, include_obsolete=plan.include_obsolete)
            hits = self.store.chunks_for_documents(docs["documents"], limit_per_doc=1)
        elif plan.strategy == "exact_then_hybrid" and (plan.doc_id or plan.prefix):
            title_filter = None
            doc_filter = plan.doc_id
            # "Find the ECR for BOM-055 Rev G" names the affected document, not
            # the ECR id. Search ECR titles instead of treating BOM-055 as the
            # target document id.
            if plan.prefix and plan.doc_id and not plan.doc_id.startswith(plan.prefix):
                title_filter = plan.doc_id
                doc_filter = None
            documents = self.store.find_documents(
                doc_id=doc_filter,
                prefix=plan.prefix if not doc_filter else None,
                title=title_filter,
                revision=None if title_filter else plan.revision,
                latest_only=plan.latest_only,
                include_obsolete=plan.include_obsolete,
            )
            if title_filter and plan.revision:
                documents = [
                    doc for doc in documents if f"Rev {plan.revision}" in str(doc["title"])
                ] or documents
            if documents:
                hits = self.store.chunks_for_documents(documents, limit_per_doc=1)
            if not hits:
                hits = self._hybrid_or_lexical(query, mode, limit, warnings)
        else:
            hits = self._hybrid_or_lexical(query, mode, limit, warnings)

        response = self.answerer.answer(query, plan, hits[:limit])
        response["mode"] = mode
        response["warnings"] = [*response.get("warnings", []), *warnings]
        return response

    def _hybrid_or_lexical(
        self, query: str, mode: str, limit: int, warnings: list[str]
    ) -> list[SearchHit]:
        try:
            if self.vector_index.exists() and mode in {"auto", "local", "hybrid"}:
                hits = self.hybrid.search(query, limit=max(limit, self.config.answer_max_chunks))
                return self.store.expand_neighbors(
                    hits[: self.config.answer_max_chunks],
                    neighbor_chunks=self.config.answer_context_neighbor_chunks,
                    parent_section_max_tokens=self.config.parent_section_max_tokens,
                )
        except Exception as exc:  # fall back to deterministic lexical search
            warnings.append(f"vector_search_fallback:{type(exc).__name__}")
        return self.store.fts_search(query, limit=limit)
