from __future__ import annotations

import io
import json
from types import SimpleNamespace

import agent.cli as chat_cli
import agent.search.cli as search_cli
from agent.config import SearchConfig


SEARCH_CALLS = []


class StubAgent:
    def invoke(self, payload):
        messages = [*payload["messages"], SimpleNamespace(content="generic reply")]
        return {"messages": messages}


class StubSearchService:
    def __init__(self, *_args, **_kwargs):
        pass

    def search(self, query, *, mode="auto", limit=16, force_strategy=None):
        SEARCH_CALLS.append(
            {
                "query": query,
                "mode": mode,
                "limit": limit,
                "force_strategy": force_strategy,
            }
        )
        return {
            "answer": f"search reply for {query}",
            "citations": [
                {
                    "doc_id": "BOM-055",
                    "revision": "G",
                    "title": "MX1 Bill of Materials",
                    "section": "metadata_inventory",
                    "filename": "BOM-055.docx",
                    "markdown_path": ".data/qms-index/normalized/BOM-055_rev-G.md",
                    "markdown_path_abs": "/repo/.data/qms-index/normalized/BOM-055_rev-G.md",
                    "source_path": "Example QMS - MedAI/BOM-055.docx",
                    "source_path_abs": "/repo/Example QMS - MedAI/BOM-055.docx",
                    "chunk_id": "BOM-055:G:metadata:0",
                }
            ],
            "query_plan": {"query": query, "strategy": "exact_then_hybrid"},
            "retrieved_documents": [],
            "warnings": [],
            "mode": mode,
            "limit": limit,
            "retrieval_backend": "local_hybrid",
            "debug_trace": {
                "original_query": query,
                "resolved_query": query,
                "retrieval_backend": "local_hybrid",
                "citation_count": 1,
            },
        }


class PathUnavailableSearchService(StubSearchService):
    def search(self, query, *, mode="auto", limit=16, force_strategy=None):
        result = super().search(
            query,
            mode=mode,
            limit=limit,
            force_strategy=force_strategy,
        )
        if "full pathname" in query.lower() or "full path" in query.lower():
            result["answer"] = "The full pathname citation is not available in the provided evidence."
        return result


def test_normalize_cli_input_strips_transcript_prefixes():
    for raw, expected in [
        ("You: what changed?", "what changed?"),
        ("User: Find BOM-055", "Find BOM-055"),
        ("Q: Find BOM-055", "Find BOM-055"),
        ("Query: Find BOM-055", "Find BOM-055"),
        ("You: User: Find BOM-055", "Find BOM-055"),
    ]:
        cleaned, trace = chat_cli._normalize_cli_input(raw)
        assert cleaned == expected
        assert trace["raw_input"] == raw
        assert trace["normalized_input"] == expected
        assert trace["input_prefix_stripped"] is True


def test_build_index_honors_configured_hash_embeddings(monkeypatch, tmp_path, capsys):
    (tmp_path / "ingest_manifest.json").write_text(
        json.dumps({"source_sha256": "hash", "documents": []}),
        encoding="utf-8",
    )
    captured = {}

    class FakeHashProvider:
        provider_name = "hash"

        def __init__(self, *args, **kwargs):
            pass

    class FakeOpenAIProvider:
        def __init__(self, *args, **kwargs):
            raise AssertionError("build_index_main should honor config.use_hash_embeddings")

    class FakeVectorIndex:
        def __init__(self, *args, **kwargs):
            pass

        def build(self, chunks, provider, *, corpus_hash):
            captured["provider_name"] = provider.provider_name
            captured["corpus_hash"] = corpus_hash
            return {"embedding_provider": provider.provider_name}

    monkeypatch.setattr(search_cli, "load_config", lambda: SearchConfig(
        index_dir=tmp_path,
        use_hash_embeddings=True,
    ))
    monkeypatch.setattr(search_cli, "HashEmbeddingProvider", FakeHashProvider)
    monkeypatch.setattr(search_cli, "OpenAIEmbeddingProvider", FakeOpenAIProvider)
    monkeypatch.setattr(search_cli, "LocalVectorIndex", FakeVectorIndex)
    monkeypatch.setattr(search_cli, "chunks_from_manifest", lambda manifest, config: [])
    monkeypatch.setattr("sys.argv", ["build-qms-index"])

    search_cli.build_index_main()

    assert captured == {"provider_name": "hash", "corpus_hash": "hash"}
    assert json.loads(capsys.readouterr().out)["embedding_provider"] == "hash"


def test_qms_followup_detection_uses_real_followup_cues():
    assert chat_cli._looks_like_followup("What revision is that?") is True
    assert chat_cli._looks_like_followup("Show me the full path citation.") is True
    assert chat_cli._looks_like_followup("Compare this document to the previous version.") is True

    assert (
        chat_cli._looks_like_followup(
            "Summarize all design review action items that are still open"
        )
        is False
    )
    assert chat_cli._looks_like_followup("Show me all risk-related documents") is False


def test_chat_cli_preserves_generic_agent_mode(monkeypatch, capsys):
    inputs = iter(["hello", "quit"])
    prompts = []
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: prompts.append(prompt) or next(inputs),
    )
    monkeypatch.setattr(chat_cli, "make_agent", lambda *_args, **_kwargs: StubAgent())
    monkeypatch.setattr("sys.argv", ["chat"])

    chat_cli.main()

    output = capsys.readouterr().out
    assert "Chat started" in output
    assert "Assistant: generic reply" in output
    assert prompts == ["You: ", "You: "]


def test_chat_cli_qms_search_mode_prints_citations(monkeypatch, capsys):
    SEARCH_CALLS.clear()
    inputs = iter(["Find BOM-055", "quit"])
    prompts = []
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: prompts.append(prompt) or next(inputs),
    )
    monkeypatch.setattr(chat_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr("sys.argv", ["chat", "--qms-search", "--mode", "local", "--limit", "2"])

    chat_cli.main()

    output = capsys.readouterr().out
    assert "Assistant: search reply for Find BOM-055" in output
    assert "Assistant (repeated): search reply for Find BOM-055" in output
    assert output.index("Assistant: search reply for Find BOM-055") < output.rindex(
        "Assistant (repeated): search reply for Find BOM-055"
    )
    assert "Citations:" in output
    assert "BOM-055 Rev G" in output
    assert prompts == ["QMS> ", "QMS> "]


def test_chat_cli_qms_strips_prompt_prefixes_before_search(monkeypatch):
    SEARCH_CALLS.clear()
    inputs = iter(
        [
            "You: what changed in the latest version from previous versions?",
            "User: Find BOM-055",
            "Query: Find BOM-055",
            "Q: Find BOM-055",
            "quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr("sys.argv", ["chat", "--qms-search", "--mode", "hybrid"])

    chat_cli.main()

    assert [call["query"] for call in SEARCH_CALLS] == [
        "what changed in the latest version from previous versions?",
        "Find BOM-055",
        "Find BOM-055",
        "Find BOM-055",
    ]


def test_chat_cli_qms_multi_turn_trace_and_full_citations(monkeypatch, capsys):
    SEARCH_CALLS.clear()
    inputs = iter(
        [
            "Find the Bill of Materials for the MX1 system",
            "What revision is that?",
            "Show me the full pathname citation.",
            "quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr(
        "sys.argv",
        [
            "chat",
            "--qms-search",
            "--mode",
            "hybrid",
            "--limit",
            "8",
            "--trace",
            "--full-citations",
        ],
    )

    chat_cli.main()

    output = capsys.readouterr().out
    assert len(SEARCH_CALLS) == 3
    assert SEARCH_CALLS[0]["query"] == "Find the Bill of Materials for the MX1 system"
    assert "BOM-055 Rev G" in SEARCH_CALLS[1]["query"]
    assert "BOM-055 Rev G" in SEARCH_CALLS[2]["query"]
    assert "Retrieval backend: local_hybrid" in output
    assert "Trace:" in output
    assert "resolved_query" in output
    assert "/repo/.data/qms-index/normalized/BOM-055_rev-G.md" in output


def test_chat_cli_clear_resets_qms_followup_context(monkeypatch, capsys):
    SEARCH_CALLS.clear()
    inputs = iter(
        [
            "Find the Bill of Materials for the MX1 system",
            "/clear",
            "What revision is that?",
            "quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr("sys.argv", ["chat", "--qms-search", "--mode", "hybrid"])

    chat_cli.main()

    output = capsys.readouterr().out
    assert "Session context cleared." in output
    assert [call["query"] for call in SEARCH_CALLS] == [
        "Find the Bill of Materials for the MX1 system",
        "What revision is that?",
    ]


def test_chat_cli_no_followup_disables_context_anchor(monkeypatch):
    SEARCH_CALLS.clear()
    inputs = iter(
        [
            "Find the Bill of Materials for the MX1 system",
            "What revision is that?",
            "quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr(
        "sys.argv",
        ["chat", "--qms-search", "--mode", "hybrid", "--no-followup"],
    )

    chat_cli.main()

    assert [call["query"] for call in SEARCH_CALLS] == [
        "Find the Bill of Materials for the MX1 system",
        "What revision is that?",
    ]


def test_chat_cli_force_strategy_is_passed_to_search_service(monkeypatch):
    SEARCH_CALLS.clear()
    inputs = iter(["Find BOM-055", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr(
        "sys.argv",
        [
            "chat",
            "--qms-search",
            "--mode",
            "hybrid",
            "--force-strategy",
            "sql_list",
        ],
    )

    chat_cli.main()

    assert SEARCH_CALLS == [
        {
            "query": "Find BOM-055",
            "mode": "hybrid",
            "limit": 16,
            "force_strategy": "sql_list",
        }
    ]


def test_chat_cli_qms_prompt_prefixed_latest_version_followup_anchors_context(
    monkeypatch, capsys
):
    SEARCH_CALLS.clear()
    inputs = iter(
        [
            "Find the Bill of Materials for the MX1 system",
            "You: what changed in the latest version from previous versions?",
            "quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr(
        "sys.argv",
        ["chat", "--qms-search", "--mode", "hybrid", "--trace"],
    )

    chat_cli.main()

    output = capsys.readouterr().out
    assert len(SEARCH_CALLS) == 2
    assert SEARCH_CALLS[1]["query"].startswith(
        "what changed in the latest version from previous versions?"
    )
    assert "You:" not in SEARCH_CALLS[1]["query"]
    assert "BOM-055 Rev G" in SEARCH_CALLS[1]["query"]
    assert '"raw_input": "You: what changed in the latest version from previous versions?"' in output
    assert '"input_prefix_stripped": true' in output


def test_chat_cli_qms_full_path_request_has_deterministic_answer(monkeypatch, capsys):
    SEARCH_CALLS.clear()
    inputs = iter(
        [
            "Find the Bill of Materials for the MX1 system",
            "Show me the full pathname citation.",
            "quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "QmsSearchService", PathUnavailableSearchService)
    monkeypatch.setattr(
        "sys.argv",
        [
            "chat",
            "--qms-search",
            "--mode",
            "hybrid",
            "--full-citations",
        ],
    )

    chat_cli.main()

    output = capsys.readouterr().out
    assert "Full citation paths are listed below for BOM-055 Rev G." in output
    assert "full pathname citation is not available" not in output
    assert "/repo/.data/qms-index/normalized/BOM-055_rev-G.md" in output


def test_chat_cli_qms_json_outputs_structured_turns(monkeypatch, capsys):
    SEARCH_CALLS.clear()
    inputs = iter(["Find BOM-055", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr(
        "sys.argv",
        ["chat", "--qms-search", "--mode", "hybrid", "--json"],
    )

    chat_cli.main()

    output = capsys.readouterr().out
    assert '"retrieval_backend": "local_hybrid"' in output
    assert '"resolved_query": "Find BOM-055"' in output


def test_search_query_cli_outputs_json(monkeypatch, capsys):
    SEARCH_CALLS.clear()
    monkeypatch.setattr(search_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr("sys.argv", ["search-qms", "Find BOM-055", "--mode", "local"])

    search_cli.query_main()

    output = capsys.readouterr().out
    assert '"answer": "search reply for Find BOM-055"' in output
    assert '"mode": "local"' in output
    assert '"retrieval_backend": "local_hybrid"' in output


def test_chat_cli_progress_is_tty_only_and_flag_gated():
    defaults = SimpleNamespace(json=False, plain=False, no_progress=False)
    assert chat_cli._should_use_progress(defaults, is_tty=True) is True
    assert chat_cli._should_use_progress(defaults, is_tty=False) is False

    assert (
        chat_cli._should_use_progress(
            SimpleNamespace(json=True, plain=False, no_progress=False),
            is_tty=True,
        )
        is False
    )
    assert (
        chat_cli._should_use_progress(
            SimpleNamespace(json=False, plain=True, no_progress=False),
            is_tty=True,
        )
        is False
    )
    assert (
        chat_cli._should_use_progress(
            SimpleNamespace(json=False, plain=False, no_progress=True),
            is_tty=True,
        )
        is False
    )


def test_chat_cli_accepts_plain_no_progress_and_raw_trace_flags(monkeypatch, capsys):
    SEARCH_CALLS.clear()
    inputs = iter(["Find BOM-055", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr(
        "sys.argv",
        [
            "chat",
            "--qms-search",
            "--mode",
            "hybrid",
            "--plain",
            "--no-progress",
            "--trace",
            "--raw-trace",
        ],
    )

    chat_cli.main()

    output = capsys.readouterr().out
    assert "Assistant: search reply for Find BOM-055" in output
    assert "Trace:" in output
    assert '"resolved_query": "Find BOM-055"' in output


def test_chat_cli_json_output_stays_parseable_with_new_flags(monkeypatch, capsys):
    SEARCH_CALLS.clear()
    inputs = iter(["Find BOM-055", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr(
        "sys.argv",
        [
            "chat",
            "--qms-search",
            "--mode",
            "hybrid",
            "--json",
            "--trace",
            "--raw-trace",
            "--full-citations",
        ],
    )

    chat_cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["retrieval_backend"] == "local_hybrid"
    assert payload["debug_trace"]["resolved_query"] == "Find BOM-055"


def test_pretty_renderer_formats_citations_and_trace():
    stream = io.StringIO()
    renderer = chat_cli.CliRenderer(pretty=True, stream=stream)
    result = StubSearchService().search("Find BOM-055", mode="hybrid", limit=8)
    result["warnings"] = ["reranker_backend:deterministic_fallback"]
    result["debug_trace"] = {
        "raw_input": "You: Find BOM-055",
        "normalized_input": "Find BOM-055",
        "input_prefix_stripped": True,
        "stripped_prefixes": ["You"],
        "original_query": "Find BOM-055",
        "resolved_query": "Find BOM-055",
        "follow_up_detected": False,
        "requested_mode": "hybrid",
        "retrieval_backend": "local_hybrid",
        "service_trace": {
            "query_plan_category": "known_item",
            "query_plan_strategy": "exact_then_hybrid",
            "retrieved_count": 1,
            "citation_count": 1,
            "warnings": ["reranker_backend:deterministic_fallback"],
        },
    }

    renderer.render_qms_result(
        result,
        trace=True,
        full_citations=True,
        raw_trace=False,
    )

    output = stream.getvalue()
    assert "Assistant" in output
    assert "Assistant (Repeated)" in output
    assert output.count("search reply for Find BOM-055") == 2
    assert output.rindex("Assistant (Repeated)") > output.rindex("Operational Trace")
    assert "Mode" in output
    assert "Retrieval backend" in output
    assert "local_hybrid" in output
    assert "Source 1" in output
    assert "BOM-055" in output
    assert "Markdown path" in output
    assert "Source path" in output
    assert "Operational Trace" in output
    assert "Input" in output
    assert "Routing" in output


def test_pretty_renderer_raw_trace_prints_json():
    stream = io.StringIO()
    renderer = chat_cli.CliRenderer(pretty=True, stream=stream)
    result = StubSearchService().search("Find BOM-055", mode="hybrid", limit=8)
    result["debug_trace"] = {"resolved_query": "Find BOM-055"}

    renderer.render_qms_result(result, trace=True, full_citations=False, raw_trace=True)

    output = stream.getvalue()
    assert "Trace:" in output
    assert '"resolved_query": "Find BOM-055"' in output
