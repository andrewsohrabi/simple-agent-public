from fastapi.testclient import TestClient

import agent.server as server
from agent.server import app


class StubSearchService:
    def __init__(self, *_args, **_kwargs):
        pass

    def search(self, query, *, mode="auto", limit=16):
        return {
            "answer": f"stub answer for {query}",
            "citations": [
                {
                    "doc_id": "BOM-055",
                    "revision": "G",
                    "title": "Bill of Materials",
                    "section": "metadata_inventory",
                    "filename": "BOM-055.docx",
                    "markdown_path": None,
                    "chunk_id": None,
                }
            ],
            "retrieved_documents": [],
            "query_plan": {"query": query},
            "warnings": [],
            "mode": mode,
            "limit": limit,
        }


def test_stats_exposes_model_and_index_configuration():
    client = TestClient(app)
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    model_config = data["model_config"]
    assert model_config["embedding_model"] == "text-embedding-3-large"
    assert model_config["embedding_dimensions"] == 3072
    assert model_config["faiss_index_type"] == "IndexFlatIP"
    assert model_config["chat_model"] == "gpt-5.5"
    assert model_config["agent_model"] == "gpt-5.5"
    assert model_config["reranker_model"] == "Qwen/Qwen3-Reranker-4B"
    reranker = data["reranker"]
    assert reranker["enabled"] is True
    assert reranker["configured_model"] == "Qwen/Qwen3-Reranker-4B"
    assert reranker["backend"] in {
        "sentence_transformers_cross_encoder",
        "deterministic_fallback",
    }
    if reranker["backend"] == "deterministic_fallback":
        assert reranker["warning"] == "real_reranker_unavailable"
        assert reranker["fallback_reason"]


def test_index_status_alias_exposes_stats():
    client = TestClient(app)
    response = client.get("/index/status")
    assert response.status_code == 200
    assert "model_config" in response.json()


def test_status_alias_exposes_stats():
    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    assert "model_config" in response.json()


def test_search_rejects_empty_query():
    client = TestClient(app)
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 400


def test_search_uses_response_contract(monkeypatch):
    monkeypatch.setattr(server, "QmsSearchService", StubSearchService)
    client = TestClient(app)
    response = client.post(
        "/search",
        json={"query": "Find BOM-055 Rev G", "mode": "local", "limit": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "stub answer for Find BOM-055 Rev G"
    assert data["citations"][0]["doc_id"] == "BOM-055"
    assert data["query_plan"]["query"] == "Find BOM-055 Rev G"
    assert data["mode"] == "local"


def test_chat_uses_search_response_contract(monkeypatch):
    monkeypatch.setattr(server, "QmsSearchService", StubSearchService)
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "How many ECRs are in the system?"}],
            "mode": "local",
            "limit": 4,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "stub answer for How many ECRs are in the system?"
    assert data["citations"][0]["doc_id"] == "BOM-055"
    assert data["query_plan"]["query"] == "How many ECRs are in the system?"
    assert data["mode"] == "local"


def test_chat_rejects_missing_user_message():
    client = TestClient(app)
    response = client.post("/chat", json={"messages": [{"role": "assistant", "content": "hi"}]})
    assert response.status_code == 400


def test_document_and_chunk_lookup_contracts():
    client = TestClient(app)
    documents_response = client.get("/documents", params={"limit": 1})
    assert documents_response.status_code == 200
    documents = documents_response.json()["documents"]
    assert documents

    doc = documents[0]
    document_response = client.get(
        f"/documents/{doc['doc_id']}",
        params={"revision": doc["revision"], "include_chunks": True, "chunk_limit": 1},
    )
    assert document_response.status_code == 200
    document_payload = document_response.json()
    assert document_payload["document"]["doc_id"] == doc["doc_id"]
    assert document_payload["chunks"]

    chunk_id = document_payload["chunks"][0]["chunk_id"]
    chunk_response = client.get(f"/chunks/{chunk_id}")
    assert chunk_response.status_code == 200
    assert chunk_response.json()["chunk"]["chunk_id"] == chunk_id


def test_missing_document_and_chunk_return_404():
    client = TestClient(app)
    assert client.get("/documents/DOES-NOT-EXIST").status_code == 404
    assert client.get("/chunks/does-not-exist").status_code == 404
