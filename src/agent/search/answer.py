from __future__ import annotations

from dataclasses import asdict

from agent.config import SearchConfig
from agent.search.answer_synthesis import (
    AnswerSynthesizer,
    OpenAIAnswerSynthesizer,
    uses_only_known_source_labels,
)
from agent.search.citations import (
    citations_for_hits,
    validate_citation_rows,
    validate_citations,
)
from agent.search.query_plan import QueryPlan
from agent.search.schema import SearchHit
from agent.search.sqlite_store import SearchStore


def _citation_label(hit: SearchHit) -> str:
    return f"{hit.doc_id} Rev {hit.revision}, {hit.section}"


class SearchAnswerer:
    def __init__(
        self,
        store: SearchStore,
        *,
        config: SearchConfig | None = None,
        synthesizer: AnswerSynthesizer | None = None,
    ):
        self.store = store
        self.config = config or SearchConfig.from_env()
        self.synthesizer = synthesizer

    def answer(self, query: str, plan: QueryPlan, hits: list[SearchHit]) -> dict[str, object]:
        citations = citations_for_hits(self.store, hits)
        citation_errors = validate_citations(self.store, citations)
        if plan.requires_count:
            docs = self._documents_for_plan(plan, limit=500)
            count = len(docs)
            doc_labels = [
                f"{doc['doc_id']} Rev {doc['revision']}" for doc in docs[:40]
            ]
            scope = plan.doc_id or plan.prefix or "matching"
            answer = (
                f"Count: {count} {scope} document revisions. "
                f"I counted records in the local SQLite metadata store, not a top-N retrieval sample. "
                f"Documents: {'; '.join(doc_labels) if doc_labels else 'none found'}."
            )
            citation_rows = self._metadata_citations(docs[:40], section="metadata_inventory")
            return {
                "answer": answer,
                "citations": citation_rows,
                "retrieved_documents": self._metadata_documents(docs[:40]),
                "query_plan": asdict(plan),
                "warnings": [*citation_errors, *validate_citation_rows(self.store, citation_rows)],
            }
        if plan.requires_revision_chain:
            return self._revision_chain_answer(plan)
        if plan.requires_list:
            docs = self._documents_for_plan(plan, limit=80)
            if docs:
                doc_labels = [
                    f"{doc['doc_id']} Rev {doc['revision']} - {doc['title']}"
                    for doc in docs[:40]
                ]
                answer = (
                    "I found these SQLite metadata records for the requested list:\n"
                    + "\n".join(f"- {label}" for label in doc_labels)
                )
                citation_rows = self._metadata_citations(docs[:40], section="metadata_inventory")
                return {
                    "answer": answer,
                    "citations": citation_rows,
                    "retrieved_documents": self._metadata_documents(docs[:40]),
                    "query_plan": asdict(plan),
                    "warnings": [
                        *citation_errors,
                        *validate_citation_rows(self.store, citation_rows),
                    ],
                }
        if not hits:
            return {
                "answer": (
                    "I could not find source-backed evidence for that request in the indexed "
                    "MedAI QMS corpus. No factual claim is made without a citation."
                ),
                "citations": [],
                "retrieved_documents": [],
                "query_plan": asdict(plan),
                "warnings": ["no_retrieval_hits"],
            }
        if plan.requires_diff:
            return self._revision_diff_answer(query, plan, hits, citations, citation_errors)

        answer = self._deterministic_retrieval_answer(hits)

        citation_rows = [asdict(citation) for citation in citations]
        warnings = list(citation_errors)
        if self.config.answer_synthesis_enabled:
            try:
                synthesizer = self.synthesizer or OpenAIAnswerSynthesizer()
                synthesized = synthesizer.synthesize(
                    query=query,
                    hits=hits,
                    citation_rows=citation_rows,
                    model=self.config.chat_model,
                    max_input_chars=self.config.answer_synthesis_max_input_chars,
                )
                if synthesized and uses_only_known_source_labels(
                    synthesized, citation_rows
                ):
                    answer = synthesized
                else:
                    warnings.append("answer_synthesis_fallback:unusable_content")
            except Exception as exc:
                warnings.append(f"answer_synthesis_fallback:{type(exc).__name__}")

        return {
            "answer": answer,
            "citations": citation_rows,
            "retrieved_documents": [
                {
                    "chunk_id": hit.chunk_id,
                    "doc_id": hit.doc_id,
                    "revision": hit.revision,
                    "title": hit.title,
                    "section": hit.section,
                    "score": hit.score,
                    "source": hit.source,
                    "metadata": hit.metadata,
                }
                for hit in hits
            ],
            "query_plan": asdict(plan),
            "warnings": warnings,
        }

    def _deterministic_retrieval_answer(self, hits: list[SearchHit]) -> str:
        top = hits[:5]
        answer_lines = [
            "I found source-backed MedAI QMS evidence for this request:",
            "",
        ]
        for hit in top:
            excerpt = " ".join(hit.text.split())[:420]
            answer_lines.append(f"- {_citation_label(hit)}: {excerpt}")
        return "\n".join(answer_lines)

    def _documents_for_plan(self, plan: QueryPlan, limit: int) -> list[dict[str, object]]:
        if plan.requires_revision_chain:
            docs: list[dict[str, object]] = []
            for row in self.store.revision_chain(
                doc_id=plan.doc_id,
                prefix=plan.prefix,
                include_obsolete=True,
                limit=limit,
            ):
                docs.extend(
                    self.store.find_documents(
                        doc_id=str(row["doc_id"]),
                        revision=str(row["revision"]),
                        latest_only=False,
                        include_obsolete=True,
                        limit=1,
                    )
                )
            return docs
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

    def _metadata_citations(
        self, docs: list[dict[str, object]], *, section: str
    ) -> list[dict[str, object]]:
        return [
            {
                "doc_id": doc["doc_id"],
                "revision": doc["revision"],
                "title": doc["title"],
                "section": section,
                "filename": doc["filename"],
                "markdown_path": doc.get("markdown_path"),
                "chunk_id": None,
            }
            for doc in docs
        ]

    def _metadata_documents(self, docs: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {
                "doc_id": doc["doc_id"],
                "revision": doc["revision"],
                "title": doc["title"],
                "section": "metadata_inventory",
                "score": 1.0,
                "source": "sqlite",
                "metadata": {
                    "filename": doc["filename"],
                    "is_latest": bool(doc["is_latest"]),
                    "is_obsolete": bool(doc["is_obsolete"]),
                    "is_signed": bool(doc["is_signed"]),
                },
            }
            for doc in docs
        ]

    def _revision_chain_answer(self, plan: QueryPlan) -> dict[str, object]:
        docs = self._documents_for_plan(plan, limit=120)
        if not docs:
            return {
                "answer": "I could not find revision-chain metadata for that request.",
                "citations": [],
                "retrieved_documents": [],
                "query_plan": asdict(plan),
                "warnings": ["no_revision_chain_records"],
            }
        by_key: dict[str, list[dict[str, object]]] = {}
        for doc in docs:
            by_key.setdefault(str(doc["canonical_doc_key"]), []).append(doc)
        lines = ["I found these revision chains in SQLite metadata:"]
        for key, group in list(by_key.items())[:20]:
            ordered = sorted(group, key=lambda doc: int(doc["revision_rank"]))
            revisions = ", ".join(
                f"Rev {doc['revision']}{' latest' if doc['is_latest'] else ''}{' obsolete' if doc['is_obsolete'] else ''}"
                for doc in ordered
            )
            lines.append(f"- {key}: {revisions}")
        citation_rows = self._metadata_citations(docs[:40], section="revision_chain")
        return {
            "answer": "\n".join(lines),
            "citations": citation_rows,
            "retrieved_documents": self._metadata_documents(docs[:40]),
            "query_plan": asdict(plan),
            "warnings": validate_citation_rows(self.store, citation_rows),
        }

    def _revision_diff_answer(
        self,
        query: str,
        plan: QueryPlan,
        hits: list[SearchHit],
        citations,
        citation_errors: list[str],
    ) -> dict[str, object]:
        by_revision: dict[str, list[SearchHit]] = {}
        for hit in hits:
            by_revision.setdefault(hit.revision, []).append(hit)
        compared = plan.compared_revisions or tuple(sorted(by_revision)[:2])
        lines = [
            "I found revision-specific evidence for this change-tracking request.",
            f"Requested comparison: {' vs '.join(compared) if compared else 'not specified'}.",
        ]
        missing = [revision for revision in compared if revision not in by_revision]
        if missing:
            lines.append(
                "The indexed corpus did not contain all requested revisions in the same "
                f"document chain; missing requested revisions: {', '.join(missing)}."
            )
        for revision in compared:
            revision_hits = by_revision.get(revision, [])
            if not revision_hits:
                continue
            excerpt = " ".join(revision_hits[0].text.split())[:420]
            lines.append(
                f"- Rev {revision}: {_citation_label(revision_hits[0])}: {excerpt}"
            )
        return {
            "answer": "\n".join(lines),
            "citations": [asdict(citation) for citation in citations],
            "retrieved_documents": [
                {
                    "chunk_id": hit.chunk_id,
                    "doc_id": hit.doc_id,
                    "revision": hit.revision,
                    "title": hit.title,
                    "section": hit.section,
                    "score": hit.score,
                    "source": hit.source,
                    "metadata": hit.metadata,
                }
                for hit in hits
            ],
            "query_plan": asdict(plan),
            "warnings": citation_errors,
        }
