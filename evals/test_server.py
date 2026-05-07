from fastapi.testclient import TestClient

from agent.server import app


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


def test_search_rejects_empty_query():
    client = TestClient(app)
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 400
