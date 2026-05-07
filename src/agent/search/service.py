from __future__ import annotations

from agent.config import SearchConfig
from agent.search.answer import SearchAnswerer
from agent.search.embeddings import HashEmbeddingProvider, OpenAIEmbeddingProvider
from agent.search.faiss_store import LocalVectorIndex
from agent.search.hybrid import HybridSearchService
from agent.search.openai_file_search import OpenAIFileSearch
from agent.search.query_plan import QueryPlan, plan_query
from agent.search.schema import SearchHit
from agent.search.sqlite_store import SearchStore


class QmsSearchService:
    def __init__(self, config: SearchConfig, *, use_hash_embeddings: bool = False):
        self.config = config
        self.store = SearchStore(config.index_dir / "qms.sqlite")
        self.vector_index = LocalVectorIndex(config.index_dir, config)
        if config.runtime_env == "production":
            self.vector_index.validate_production_ready()
        self.embedding_provider = (
            HashEmbeddingProvider(config.embedding_dimensions, model=config.embedding_model)
            if use_hash_embeddings
            else OpenAIEmbeddingProvider(config)
        )
        self.hybrid = HybridSearchService(
            self.store, self.vector_index, self.embedding_provider, config
        )
        self.answerer = SearchAnswerer(self.store, config=config)
        self.hosted_search = OpenAIFileSearch(config)

    def search(self, query: str, *, mode: str = "auto", limit: int = 16) -> dict[str, object]:
        plan = plan_query(query)
        hits: list[SearchHit] = []
        warnings: list[str] = []

        if plan.strategy == "sql_count":
            documents = self._documents_for_sql_plan(plan, limit=max(limit, 50))
            hits = self.store.chunks_for_documents(documents, limit_per_doc=1)
        elif plan.strategy == "sql_list":
            documents = self._documents_for_sql_plan(plan, limit=max(limit, 50))
            hits = self.store.chunks_for_documents(documents, limit_per_doc=1)
        elif plan.strategy == "revision_chain":
            documents = self._revision_chain_documents(plan, limit=max(limit, 100))
            hits = self.store.chunks_for_documents(documents, limit_per_doc=1)
        elif plan.strategy == "revision_diff":
            hits = self._revision_diff_hits(plan, limit, warnings)
        elif plan.strategy == "multi_hop":
            hits = self._multi_hop_hits(query, mode, limit, warnings)
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

        hits = self._apply_obsolete_scope(plan, hits, warnings)
        response = self.answerer.answer(query, plan, hits[:limit])
        response["mode"] = mode
        response["warnings"] = [*response.get("warnings", []), *warnings]
        return response

    def _documents_for_sql_plan(self, plan, limit: int) -> list[dict[str, object]]:
        if plan.doc_id or plan.prefix:
            return self.store.find_documents(
                doc_id=plan.doc_id,
                prefix=plan.prefix if not plan.doc_id else None,
                revision=plan.revision,
                latest_only=plan.latest_only,
                include_obsolete=plan.include_obsolete,
                limit=limit,
            )

        lower = plan.query.lower()
        clauses: list[str] = []
        values: list[object] = []
        if "obsolete" in lower:
            clauses.append("is_obsolete = 1")
        elif not plan.include_obsolete:
            clauses.append("is_obsolete = 0")
        if "signed" in lower:
            clauses.append("is_signed = 1")
        if plan.latest_only:
            clauses.append("is_latest = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT *
            FROM documents
            {where}
            ORDER BY prefix, doc_id, revision_rank DESC
            LIMIT ?
        """
        values.append(limit)
        with self.store.connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [dict(row) for row in rows]

    def _revision_chain_documents(self, plan, limit: int) -> list[dict[str, object]]:
        chain = self.store.revision_chain(
            doc_id=plan.doc_id,
            prefix=plan.prefix,
            include_obsolete=True,
            limit=limit,
        )
        documents: list[dict[str, object]] = []
        for row in chain:
            matches = self.store.find_documents(
                doc_id=str(row["doc_id"]),
                revision=str(row["revision"]),
                latest_only=False,
                include_obsolete=True,
                limit=1,
            )
            documents.extend(matches)
        return documents

    def _revision_diff_hits(self, plan, limit: int, warnings: list[str]) -> list[SearchHit]:
        documents = self.store.find_documents(
            doc_id=plan.doc_id,
            prefix=plan.prefix,
            latest_only=False,
            include_obsolete=True,
            limit=100,
        )
        if not documents:
            warnings.append("revision_diff_no_document_candidates")
            return self._hybrid_or_lexical(plan.query, "local", limit, warnings)
        compared = set(plan.compared_revisions or ())
        selected: list[dict[str, object]] = []
        if compared:
            by_key: dict[str, list[dict[str, object]]] = {}
            for doc in documents:
                by_key.setdefault(str(doc["canonical_doc_key"]), []).append(doc)
            for group in by_key.values():
                revisions = {str(doc["revision"]) for doc in group}
                if compared.issubset(revisions):
                    selected = [
                        doc for doc in group if str(doc["revision"]) in compared
                    ]
                    break
            if not selected:
                warnings.append(
                    "revision_diff_exact_pair_not_found:"
                    + ",".join(sorted(compared))
                )
        if not selected:
            selected = documents[: max(2, min(limit, 6))]
        return self.store.chunks_for_documents(selected, limit_per_doc=2)

    def _multi_hop_hits(
        self, query: str, mode: str, limit: int, warnings: list[str]
    ) -> list[SearchHit]:
        base_hits = self._hybrid_or_lexical(query, mode, limit, warnings)
        expanded = list(base_hits)
        seen = {hit.chunk_id for hit in expanded}
        followed = 0
        for hit in base_hits[:5]:
            for reference in self.store.references_from(hit.doc_id, hit.revision)[:8]:
                target_doc_id = str(reference["target_doc_id"])
                documents = self.store.find_documents(
                    doc_id=target_doc_id,
                    latest_only=True,
                    include_obsolete=False,
                    limit=1,
                )
                for target_hit in self.store.chunks_for_documents(documents, limit_per_doc=1):
                    if target_hit.chunk_id in seen:
                        continue
                    seen.add(target_hit.chunk_id)
                    followed += 1
                    expanded.append(
                        SearchHit(
                            chunk_id=target_hit.chunk_id,
                            doc_id=target_hit.doc_id,
                            revision=target_hit.revision,
                            title=target_hit.title,
                            section=target_hit.section,
                            text=target_hit.text,
                            score=target_hit.score * 0.9,
                            source="reference_follow",
                            metadata={
                                **target_hit.metadata,
                                "referenced_from": f"{hit.doc_id} Rev {hit.revision}",
                                "reference_text": reference["reference_text"],
                            },
                        )
                    )
        warnings.append(f"references_followed:{followed}")
        return expanded[: max(limit, self.config.answer_max_chunks)]

    def _hybrid_or_lexical(
        self, query: str, mode: str, limit: int, warnings: list[str]
    ) -> list[SearchHit]:
        if mode in {"hosted", "auto"} and self.hosted_search.is_available():
            try:
                hosted_hits = self.hosted_search.search(query, self.store, limit=limit)
                if hosted_hits:
                    if mode == "auto":
                        warnings.append("hosted_file_search_used")
                    return hosted_hits
                warnings.append("hosted_file_search_no_hits")
            except Exception as exc:
                warnings.append(f"hosted_file_search_fallback:{type(exc).__name__}")
        elif mode == "hosted":
            warnings.append("hosted_file_search_unavailable")
        try:
            if self.vector_index.exists() and mode in {"auto", "local", "hybrid", "hosted"}:
                hits = self.hybrid.search(query, limit=max(limit, self.config.answer_max_chunks))
                return self.store.expand_neighbors(
                    hits[: self.config.answer_max_chunks],
                    neighbor_chunks=self.config.answer_context_neighbor_chunks,
                    parent_section_max_tokens=self.config.parent_section_max_tokens,
                )
        except Exception as exc:  # fall back to deterministic lexical search
            warnings.append(f"vector_search_fallback:{type(exc).__name__}")
        return self.store.fts_search(query, limit=limit)

    def _apply_obsolete_scope(
        self, plan: QueryPlan, hits: list[SearchHit], warnings: list[str]
    ) -> list[SearchHit]:
        annotated = self._annotate_obsolete_metadata(hits)
        if plan.include_obsolete:
            return annotated
        filtered = [
            hit
            for hit in annotated
            if _metadata_obsolete_flag(hit.metadata.get("is_obsolete")) is not True
        ]
        removed = len(annotated) - len(filtered)
        if removed:
            warnings.append(f"obsolete_hits_filtered:{removed}")
        return filtered

    def _annotate_obsolete_metadata(self, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        statuses = self._document_obsolete_statuses(hits)
        annotated: list[SearchHit] = []
        for hit in hits:
            key = (hit.doc_id.upper(), hit.revision.upper())
            is_obsolete = statuses.get(key)
            if is_obsolete is None:
                is_obsolete = _metadata_obsolete_flag(hit.metadata.get("is_obsolete"))
            if is_obsolete is None:
                annotated.append(hit)
                continue
            annotated.append(
                SearchHit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    revision=hit.revision,
                    title=hit.title,
                    section=hit.section,
                    text=hit.text,
                    score=hit.score,
                    source=hit.source,
                    metadata={**hit.metadata, "is_obsolete": is_obsolete},
                )
            )
        return annotated

    def _document_obsolete_statuses(
        self, hits: list[SearchHit]
    ) -> dict[tuple[str, str], bool]:
        keys = sorted({(hit.doc_id.upper(), hit.revision.upper()) for hit in hits})
        statuses: dict[tuple[str, str], bool] = {}
        with self.store.connect() as conn:
            for doc_id, revision in keys:
                row = conn.execute(
                    """
                    SELECT is_obsolete
                    FROM documents
                    WHERE doc_id = ? AND revision = ?
                    """,
                    (doc_id, revision),
                ).fetchone()
                if row is not None:
                    statuses[(doc_id, revision)] = bool(row["is_obsolete"])
        return statuses


def _metadata_obsolete_flag(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return None
