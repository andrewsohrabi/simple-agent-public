from __future__ import annotations

from types import SimpleNamespace

import agent.cli as chat_cli
import agent.search.cli as search_cli


class StubAgent:
    def invoke(self, payload):
        messages = [*payload["messages"], SimpleNamespace(content="generic reply")]
        return {"messages": messages}


class StubSearchService:
    def __init__(self, *_args, **_kwargs):
        pass

    def search(self, query, *, mode="auto", limit=16):
        return {
            "answer": f"search reply for {query}",
            "citations": [
                {
                    "doc_id": "BOM-055",
                    "revision": "G",
                    "section": "metadata_inventory",
                }
            ],
            "query_plan": {"query": query},
            "retrieved_documents": [],
            "warnings": [],
            "mode": mode,
            "limit": limit,
        }


def test_chat_cli_preserves_generic_agent_mode(monkeypatch, capsys):
    inputs = iter(["hello", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "make_agent", lambda *_args, **_kwargs: StubAgent())
    monkeypatch.setattr("sys.argv", ["chat"])

    chat_cli.main()

    output = capsys.readouterr().out
    assert "Chat started" in output
    assert "Assistant: generic reply" in output


def test_chat_cli_qms_search_mode_prints_citations(monkeypatch, capsys):
    inputs = iter(["Find BOM-055", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(chat_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr("sys.argv", ["chat", "--qms-search", "--mode", "local", "--limit", "2"])

    chat_cli.main()

    output = capsys.readouterr().out
    assert "Assistant: search reply for Find BOM-055" in output
    assert "Citations:" in output
    assert "BOM-055 Rev G" in output


def test_search_query_cli_outputs_json(monkeypatch, capsys):
    monkeypatch.setattr(search_cli, "QmsSearchService", StubSearchService)
    monkeypatch.setattr("sys.argv", ["search-qms", "Find BOM-055", "--mode", "local"])

    search_cli.query_main()

    output = capsys.readouterr().out
    assert '"answer": "search reply for Find BOM-055"' in output
    assert '"mode": "local"' in output
