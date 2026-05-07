from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from agent.config import SearchConfig
from agent.search.answer import SearchAnswerer
from agent.search.embeddings import HashEmbeddingProvider, OpenAIEmbeddingProvider
from agent.search.faiss_store import LocalVectorIndex
from agent.search.hybrid import HybridSearchService
from agent.search.openai_file_search import OpenAIFileSearch
from agent.search.query_expansion import expand_query
from agent.search.query_plan import QueryPlan, plan_query
from agent.search.schema import SearchHit
from agent.search.sqlite_store import SearchStore


@dataclass(frozen=True)
class RetrievalOutcome:
    hits: list[SearchHit]
    backend: str
    trace: dict[str, object] | None = None


class QmsSearchService:
    def __init__(self, config: SearchConfig, *, use_hash_embeddings: bool = False):
        self.config = config
        self.store = SearchStore(config.index_dir / "qms.sqlite")
        self.vector_index = LocalVectorIndex(config.index_dir, config)
        self.artifact_validation = self.vector_index.validation()
        if config.runtime_env == "production" and not self.artifact_validation.get("ok", False):
            errors = self.artifact_validation.get("errors", [])
            detail = "; ".join(str(error) for error in errors) or "unknown artifact validation error"
            raise RuntimeError(f"production search index contract validation failed: {detail}")
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

    def search(
        self,
        query: str,
        *,
        mode: str = "auto",
        limit: int = 16,
        force_strategy: str | None = None,
    ) -> dict[str, object]:
        mode = _normalize_mode(mode)
        plan = plan_query(query)
        if force_strategy:
            plan = _forced_plan(plan, force_strategy)
        hits: list[SearchHit] = []
        warnings: list[str] = []
        retrieval_backend = "unknown"
        retrieval_trace: dict[str, object] | None = None
        if self.artifact_validation.get("requires_rebuild"):
            warnings.append("index_artifact_validation_requires_rebuild")

        if plan.strategy == "sql_count":
            documents = self._documents_for_sql_plan(plan, limit=max(limit, 50))
            hits = self.store.chunks_for_documents(documents, limit_per_doc=1)
            retrieval_backend = "sql_inventory"
        elif plan.strategy == "sql_list":
            documents = self._documents_for_sql_plan(plan, limit=max(limit, 50))
            hits = self.store.chunks_for_documents(documents, limit_per_doc=1)
            retrieval_backend = "sql_inventory"
        elif plan.strategy == "revision_chain":
            documents = self._revision_chain_documents(plan, limit=max(limit, 100))
            hits = self.store.chunks_for_documents(documents, limit_per_doc=1)
            retrieval_backend = "revision_chain"
        elif plan.strategy == "revision_diff":
            outcome = self._revision_diff_hits(plan, limit, warnings)
            hits = outcome.hits
            retrieval_backend = outcome.backend
            retrieval_trace = outcome.trace
        elif plan.strategy == "multi_hop":
            outcome = self._multi_hop_hits(query, mode, limit, warnings, plan=plan)
            hits = outcome.hits
            retrieval_backend = outcome.backend
            retrieval_trace = outcome.trace
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
                retrieval_backend = "sqlite_metadata"
            if not hits:
                outcome = self._hybrid_or_lexical(query, mode, limit, warnings, plan=plan)
                hits = outcome.hits
                retrieval_backend = outcome.backend
                retrieval_trace = outcome.trace
        else:
            outcome = self._hybrid_or_lexical(query, mode, limit, warnings, plan=plan)
            hits = outcome.hits
            retrieval_backend = outcome.backend
            retrieval_trace = outcome.trace

        hits = self._apply_obsolete_scope(plan, hits, warnings)
        response = self.answerer.answer(query, plan, hits[:limit])
        response["mode"] = mode
        response["warnings"] = [*response.get("warnings", []), *warnings]
        response["retrieval_backend"] = retrieval_backend
        response["debug_trace"] = {
            "requested_mode": mode,
            "retrieval_backend": retrieval_backend,
            "query_plan_strategy": plan.strategy,
            "query_plan_category": plan.category,
            "retrieved_count": len(response.get("retrieved_documents", [])),
            "citation_count": len(response.get("citations", [])),
            "warnings": response["warnings"],
            "artifact_validation": {
                "ok": self.artifact_validation.get("ok"),
                "requires_rebuild": self.artifact_validation.get("requires_rebuild"),
                "errors": self.artifact_validation.get("errors", []),
                "warnings": self.artifact_validation.get("warnings", []),
            },
        }
        if force_strategy:
            response["debug_trace"]["force_strategy"] = force_strategy
        if retrieval_trace:
            response["debug_trace"]["retrieval_trace"] = retrieval_trace
        return response

    def _documents_for_sql_plan(self, plan, limit: int) -> list[dict[str, object]]:
        if plan.intent == "risk_related_inventory":
            return self.store.risk_related_documents(
                latest_only=plan.latest_only,
                include_obsolete=plan.include_obsolete,
                limit=max(limit, 500),
            )
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

    def _revision_diff_hits(
        self, plan, limit: int, warnings: list[str]
    ) -> RetrievalOutcome:
        documents = self.store.find_documents(
            doc_id=plan.doc_id,
            prefix=plan.prefix,
            latest_only=False,
            include_obsolete=True,
            limit=100,
        )
        if not documents:
            warnings.append("revision_diff_no_document_candidates")
            return self._hybrid_or_lexical(plan.query, "local", limit, warnings, plan=plan)
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
        return RetrievalOutcome(
            self.store.chunks_for_documents(selected, limit_per_doc=2),
            "revision_diff",
        )

    def _multi_hop_hits(
        self,
        query: str,
        mode: str,
        limit: int,
        warnings: list[str],
        *,
        plan: QueryPlan | None = None,
    ) -> RetrievalOutcome:
        base = self._hybrid_or_lexical(query, mode, limit, warnings, plan=plan)
        expanded = list(base.hits)
        seen = {hit.chunk_id for hit in expanded}
        followed = 0
        for hit in base.hits[:5]:
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
                            evidence_type=target_hit.evidence_type,
                            support_level=target_hit.support_level,
                            table_index=target_hit.table_index,
                            row_start=target_hit.row_start,
                            row_end=target_hit.row_end,
                            heading_path=target_hit.heading_path,
                            columns=target_hit.columns,
                            row_cells=target_hit.row_cells,
                        )
                    )
        warnings.append(f"references_followed:{followed}")
        return RetrievalOutcome(
            expanded[: max(limit, self.config.answer_max_chunks)],
            f"multi_hop:{base.backend}",
        )

    def _hybrid_or_lexical(
        self,
        query: str,
        mode: str,
        limit: int,
        warnings: list[str],
        *,
        plan: QueryPlan | None = None,
    ) -> RetrievalOutcome:
        expansion = expand_query(query)
        if mode in {"hosted", "auto"} and self.hosted_search.is_available():
            try:
                hosted_hits = self.hosted_search.search(query, self.store, limit=limit)
                if hosted_hits:
                    if mode == "auto":
                        warnings.append("hosted_file_search_used")
                    return RetrievalOutcome(
                        hosted_hits,
                        "hosted_file_search",
                        trace={
                            "hosted_file_search": {
                                "used": True,
                                "returned": len(hosted_hits),
                            },
                            "query_expansion": _expansion_trace(expansion),
                        },
                    )
                warnings.append("hosted_file_search_no_hits")
            except Exception as exc:
                warnings.append(f"hosted_file_search_fallback:{type(exc).__name__}")
        elif mode == "hosted":
            warnings.append("hosted_file_search_unavailable")
        try:
            if self.vector_index.exists() and mode in {"auto", "local", "hybrid", "hosted"}:
                lexical_priority = _uses_lexical_priority(plan)
                retrieval_query = (
                    expansion.expanded_query
                    if expansion.terms
                    else expansion.normalized_query
                )
                traced = self.hybrid.search_with_trace(
                    retrieval_query,
                    limit=max(
                        limit,
                        self.config.answer_max_chunks,
                        self.config.reranker_top_n_candidates,
                    ),
                    lexical_weight=2.0 if lexical_priority else 1.0,
                    vector_weight=1.0,
                )
                hits = traced.hits
                trace: dict[str, object] = {
                    "query_expansion": _expansion_trace(expansion),
                    "hybrid": asdict(traced.trace),
                    "retrieval_query": retrieval_query,
                }
                if lexical_priority:
                    preserved = []
                    for lexical_query in _preservation_queries(query, expansion):
                        preserved.extend(self.store.fts_search(lexical_query, limit=5))
                    hits = _merge_hits([*preserved, *hits])
                    trace["lexical_preservation"] = {
                        "queries": _preservation_queries(query, expansion),
                        "preserved_count": len(preserved),
                    }
                hits = _dedupe_hits(hits, max_per_section=2 if lexical_priority else 3)
                trace["post_dedupe_count"] = len(hits)
                expanded_hits = self.store.expand_neighbors(
                    hits[: self.config.answer_max_chunks],
                    neighbor_chunks=self.config.answer_context_neighbor_chunks,
                    parent_section_max_tokens=self.config.parent_section_max_tokens,
                )
                trace["neighbor_expansion_count"] = len(expanded_hits)
                return RetrievalOutcome(
                    expanded_hits,
                    "local_hybrid",
                    trace=trace,
                )
            warnings.append("local_vector_index_unavailable")
        except Exception as exc:  # fall back to deterministic lexical search
            warnings.append(f"vector_search_fallback:{type(exc).__name__}")
        fts_hits = _dedupe_hits(self.store.fts_search(expansion.expanded_query, limit=limit))
        return RetrievalOutcome(
            fts_hits,
            "local_fts",
            trace={
                "query_expansion": _expansion_trace(expansion),
                "fallback": "local_fts",
                "returned": len(fts_hits),
            },
        )

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
                    evidence_type=hit.evidence_type,
                    support_level=hit.support_level,
                    table_index=hit.table_index,
                    row_start=hit.row_start,
                    row_end=hit.row_end,
                    heading_path=hit.heading_path,
                    columns=hit.columns,
                    row_cells=hit.row_cells,
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


def _forced_plan(plan: QueryPlan, strategy: str) -> QueryPlan:
    normalized = strategy.strip().lower()
    if normalized not in {
        "hybrid",
        "exact_then_hybrid",
        "sql_count",
        "sql_list",
        "revision_chain",
        "revision_diff",
        "multi_hop",
    }:
        raise ValueError(
            "--force-strategy must be one of hybrid, exact_then_hybrid, "
            "sql_count, sql_list, revision_chain, revision_diff, or multi_hop"
        )
    warnings = [*plan.warnings, f"force_strategy:{normalized}"]
    if normalized == "sql_count":
        return replace(
            plan,
            strategy=normalized,
            requires_count=True,
            requires_list=False,
            requires_revision_chain=False,
            requires_diff=False,
            warnings=warnings,
        )
    if normalized == "sql_list":
        return replace(
            plan,
            strategy=normalized,
            requires_count=False,
            requires_list=True,
            requires_revision_chain=False,
            requires_diff=False,
            warnings=warnings,
        )
    if normalized == "revision_chain":
        return replace(
            plan,
            category="revision_diff",
            strategy=normalized,
            latest_only=False,
            include_obsolete=True,
            requires_count=False,
            requires_list=True,
            requires_revision_chain=True,
            requires_diff=False,
            warnings=warnings,
        )
    if normalized == "revision_diff":
        return replace(
            plan,
            category="revision_diff",
            strategy=normalized,
            latest_only=False,
            include_obsolete=True,
            requires_count=False,
            requires_list=False,
            requires_revision_chain=False,
            requires_diff=True,
            warnings=warnings,
        )
    return replace(
        plan,
        strategy=normalized,
        requires_count=False,
        requires_list=False,
        requires_revision_chain=False,
        requires_diff=False,
        warnings=warnings,
    )


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "auto").strip().lower()
    if normalized not in {"auto", "hosted", "hybrid", "local"}:
        raise ValueError(
            "mode must be one of auto, hosted, hybrid, or local; "
            f"got {mode!r}"
        )
    return normalized


def _uses_lexical_priority(plan: QueryPlan | None) -> bool:
    if plan is None:
        return False
    if plan.category in {"extraction", "known_item", "compliance"}:
        return True
    return plan.intent in {
        "510k_summary_location",
        "electrical_safety_acceptance",
        "open_design_review_actions",
        "electrical_leakage_trace",
        "third_party_report_mapping",
    }


def _merge_hits(hits: list[SearchHit]) -> list[SearchHit]:
    merged: list[SearchHit] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        merged.append(hit)
    return merged


def _expansion_trace(expansion) -> dict[str, object]:
    return {
        "original_query": expansion.original_query,
        "normalized_query": expansion.normalized_query,
        "expanded_query": expansion.expanded_query,
        "terms": list(expansion.terms),
        "matched_groups": list(expansion.matched_groups),
    }


def _preservation_queries(query: str, expansion) -> list[str]:
    queries = [query]
    if expansion.normalized_query and expansion.normalized_query != query:
        queries.append(expansion.normalized_query)
    for term in expansion.terms[:8]:
        queries.append(term)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_hits(hits: list[SearchHit], *, max_per_section: int = 3) -> list[SearchHit]:
    deduped: list[SearchHit] = []
    seen_chunks: set[str] = set()
    section_counts: dict[tuple[str, str, str], int] = {}
    for hit in hits:
        if hit.chunk_id in seen_chunks:
            continue
        section_key = (hit.doc_id, hit.revision, hit.section)
        count = section_counts.get(section_key, 0)
        if count >= max_per_section:
            continue
        seen_chunks.add(hit.chunk_id)
        section_counts[section_key] = count + 1
        deduped.append(hit)
    return deduped
