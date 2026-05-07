from __future__ import annotations

import json

from agent.config import SearchConfig
from agent.search.answer import SearchAnswerer
from agent.search.query_plan import QueryPlan
from agent.search.schema import SearchHit
from agent.search.sqlite_store import SearchStore


def _store(tmp_path) -> SearchStore:
    store = SearchStore(tmp_path / "qms.sqlite")
    store.initialize()
    body = (
        "BOM-055 Rev G is the latest active bill of materials. "
        "It lists MX1 assembly material controls and approved component records."
    )
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO documents
            (doc_id, revision, prefix, title, revision_rank, canonical_doc_key,
             is_latest, is_signed, is_obsolete, filename, source_path, software_version,
             markdown_path, sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BOM-055",
                "G",
                "BOM",
                "MX1 Bill of Materials",
                7,
                "BOM-055",
                1,
                1,
                0,
                "BOM-055_rev-G.docx",
                "BOM-055_rev-G.docx",
                None,
                "markdown/BOM-055_rev-G.md",
                "sha-BOM-055-G",
            ),
        )
        conn.execute(
            """
            INSERT INTO revisions
            (canonical_doc_key, doc_id, revision, revision_rank, is_latest, is_obsolete, filename)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("BOM-055", "BOM-055", "G", 7, 1, 0, "BOM-055_rev-G.docx"),
        )
        conn.execute(
            """
            INSERT INTO chunks
            (chunk_id, doc_id, revision, title, section, ordinal, text, search_text,
             parent_section_id, kind, token_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BOM-055:G:metadata:0",
                "BOM-055",
                "G",
                "MX1 Bill of Materials",
                "Document Metadata",
                0,
                body,
                body,
                "BOM-055:G:metadata",
                "metadata",
                len(body.split()),
                json.dumps({"filename": "BOM-055_rev-G.docx"}),
            ),
        )
    return store


def _hit() -> SearchHit:
    return SearchHit(
        chunk_id="BOM-055:G:metadata:0",
        doc_id="BOM-055",
        revision="G",
        title="MX1 Bill of Materials",
        section="Document Metadata",
        text="BOM-055 Rev G is the latest active bill of materials for MX1.",
        score=0.98,
        source="hybrid",
        metadata={"is_obsolete": False},
    )


def _plan(**overrides) -> QueryPlan:
    values = {
        "category": "known_item",
        "strategy": "hybrid",
        "query": "Summarize BOM-055",
    }
    values.update(overrides)
    return QueryPlan(**values)


class RecordingSynthesizer:
    def __init__(self, text: str | None = None, error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls: list[dict[str, object]] = []

    def synthesize(self, *, query, hits, citation_rows, model, max_input_chars):
        self.calls.append(
            {
                "query": query,
                "hits": hits,
                "citation_rows": citation_rows,
                "model": model,
                "max_input_chars": max_input_chars,
            }
        )
        if self.error:
            raise self.error
        return self.text


def test_synthesized_answer_uses_config_chat_model_and_keeps_local_citations(tmp_path):
    store = _store(tmp_path)
    synthesizer = RecordingSynthesizer(
        "BOM-055 Rev G is the active bill of materials for MX1. [S1]"
    )
    answerer = SearchAnswerer(store, config=SearchConfig(), synthesizer=synthesizer)

    result = answerer.answer("Summarize BOM-055", _plan(), [_hit()])

    assert result["answer"] == "BOM-055 Rev G is the active bill of materials for MX1. [S1]"
    assert result["citations"] == [
        {
            "doc_id": "BOM-055",
            "revision": "G",
            "title": "MX1 Bill of Materials",
            "section": "Document Metadata",
            "filename": "BOM-055_rev-G.docx",
            "markdown_path": "markdown/BOM-055_rev-G.md",
            "chunk_id": "BOM-055:G:metadata:0",
        }
    ]
    assert result["warnings"] == []
    assert synthesizer.calls[0]["model"] == "gpt-5.5"
    assert synthesizer.calls[0]["max_input_chars"] == 12000


def test_synthesis_failure_falls_back_to_deterministic_answer_with_warning(tmp_path):
    store = _store(tmp_path)
    synthesizer = RecordingSynthesizer(error=RuntimeError("network unavailable"))
    answerer = SearchAnswerer(store, config=SearchConfig(), synthesizer=synthesizer)

    result = answerer.answer("Summarize BOM-055", _plan(), [_hit()])

    assert "I found source-backed MedAI QMS evidence" in result["answer"]
    assert "BOM-055 Rev G, Document Metadata" in result["answer"]
    assert "answer_synthesis_fallback:RuntimeError" in result["warnings"]
    assert result["citations"][0]["doc_id"] == "BOM-055"


def test_count_answer_path_avoids_answer_synthesis(tmp_path):
    store = _store(tmp_path)
    synthesizer = RecordingSynthesizer("Should not be used. [S1]")
    answerer = SearchAnswerer(store, config=SearchConfig(), synthesizer=synthesizer)

    result = answerer.answer(
        "How many BOM-055 records are present?",
        _plan(strategy="sql_count", requires_count=True, doc_id="BOM-055"),
        [_hit()],
    )

    assert result["answer"].startswith("Count: 1 BOM-055 document revisions.")
    assert synthesizer.calls == []
