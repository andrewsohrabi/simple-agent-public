from __future__ import annotations

from dataclasses import asdict

from agent.search.citations import citations_for_hits
from agent.search.query_plan import QueryPlan
from agent.search.schema import SearchHit
from agent.search.sqlite_store import SearchStore


def _citation_label(hit: SearchHit) -> str:
    return f"{hit.doc_id} Rev {hit.revision}, {hit.section}"


class SearchAnswerer:
    def __init__(self, store: SearchStore):
        self.store = store

    def answer(self, query: str, plan: QueryPlan, hits: list[SearchHit]) -> dict[str, object]:
        citations = citations_for_hits(self.store, hits)
        if plan.requires_count and plan.prefix:
            count_result = self.store.count_by_prefix(
                plan.prefix, include_obsolete=plan.include_obsolete
            )
            docs = count_result["documents"]
            doc_labels = [
                f"{doc['doc_id']} Rev {doc['revision']}" for doc in docs[:40]
            ]
            answer = (
                f"Count: {count_result['count']} {plan.prefix} document revisions. "
                f"I counted records in the local SQLite metadata store, not a top-N retrieval sample. "
                f"Documents: {'; '.join(doc_labels) if doc_labels else 'none found'}."
            )
            citation_rows = [
                {
                    "doc_id": doc["doc_id"],
                    "revision": doc["revision"],
                    "title": doc["title"],
                    "section": "metadata_inventory",
                    "filename": doc["filename"],
                    "markdown_path": doc.get("markdown_path"),
                    "chunk_id": None,
                }
                for doc in docs[:40]
            ]
            return {
                "answer": answer,
                "citations": citation_rows,
                "retrieved_documents": [
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
                        },
                    }
                    for doc in docs[:40]
                ],
                "query_plan": asdict(plan),
                "warnings": [],
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

        top = hits[:5]
        answer_lines = [
            "I found source-backed MedAI QMS evidence for this request:",
            "",
        ]
        for hit in top:
            excerpt = " ".join(hit.text.split())[:420]
            answer_lines.append(f"- {_citation_label(hit)}: {excerpt}")
        answer = "\n".join(answer_lines)

        return {
            "answer": answer,
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
            "warnings": [],
        }
