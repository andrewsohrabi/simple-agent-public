from fastapi.testclient import TestClient

import agent.server as server


client = TestClient(server.app)


def test_health_endpoint_reports_actual_checks(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MEMORY_ROOT", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    response = client.get("/health")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["serverReachable"] is True
    assert payload["memoryStore"]["writable"] is True
    assert payload["providers"]["anthropic"]["configured"] is False
    assert "do not validate" in payload["note"]


def test_demo_users_endpoint_lists_seeded_paths():
    response = client.get("/demo-users")

    assert response.status_code == 200
    payload = response.json()
    assert payload["defaultDemoUser"] == "blank_demo"
    assert {user["key"] for user in payload["demoUsers"]} >= {
        "blank_demo",
        "regulatory_lead",
        "personal_preferences",
        "style_constrained",
    }


def test_load_demo_user_endpoint_seeds_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MEMORY_ROOT", tmp_path)

    response = client.post("/demo-users/load", json={"demoUser": "personal_preferences"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["userId"] == "demo_personal_preferences"
    assert payload["facts"]["preferences"]["preferred_fruit"]["value"] == "mango"
    assert payload["factsEvents"] == []
    assert payload["factsEventsPath"] == str(
        tmp_path / "demo_personal_preferences" / "facts_events.jsonl"
    )
    assert "Andrew" in payload["summary"]["summary"]
    assert payload["promptPreviews"]["none"] is None
    assert "Long-Term Memory:" in payload["promptPreviews"]["facts"]
    assert "Long-Term Memory:" in payload["promptPreviews"]["summary"]
    assert (tmp_path / "demo_personal_preferences" / "facts.json").exists()
    assert (tmp_path / "demo_personal_preferences" / "summary.json").exists()


def test_chat_endpoint_returns_reply_and_memory(monkeypatch):
    monkeypatch.setattr(
        server,
        "run_chat_turn",
        lambda **kwargs: {
            "reply": "Stored.",
            "memory": {"profile": {"name": {"value": "Andrew"}}},
            "memory_path": "memory_store/demo_user/facts.json",
            "model_used": "anthropic:test-model",
            "system_prompt": "Long-Term Memory:\n- name: Andrew",
            "messages": kwargs["messages"],
        },
    )
    monkeypatch.setattr(
        server,
        "read_facts_events_snapshot",
        lambda memory_type, memory_root, user_id: (
            [
                {
                    "timestamp": "2026-04-10T12:34:56Z",
                    "category": "profile",
                    "key": "name",
                    "old_value": None,
                    "new_value": "Andrew",
                    "source": "My name is Andrew.",
                }
            ],
            server.Path("memory_store/demo_user/facts_events.jsonl"),
        ),
    )

    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "My name is Andrew."}],
            "memoryType": "facts",
            "userId": "demo_user",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "Stored.",
        "memory": {"profile": {"name": {"value": "Andrew"}}},
        "memoryPath": "memory_store/demo_user/facts.json",
        "factsEvents": [
            {
                "timestamp": "2026-04-10T12:34:56Z",
                "category": "profile",
                "key": "name",
                "old_value": None,
                "new_value": "Andrew",
                "source": "My name is Andrew.",
            }
        ],
        "factsEventsPath": "memory_store/demo_user/facts_events.jsonl",
        "modelUsed": "anthropic:test-model",
        "promptPreview": "Long-Term Memory:\n- name: Andrew",
    }


def test_get_memory_endpoint_returns_snapshot(monkeypatch):
    monkeypatch.setattr(
        server,
        "read_memory_snapshot",
        lambda memory_type, memory_root, user_id: (
            {"summary": "User is Andrew.", "updated_at": "2026-04-10T12:00:00Z"},
            server.Path("memory_store/demo_user/summary.json"),
        ),
    )
    monkeypatch.setattr(
        server,
        "build_system_prompt_preview",
        lambda **kwargs: "Long-Term Memory:\nUser is Andrew.",
    )
    monkeypatch.setattr(
        server,
        "read_facts_events_snapshot",
        lambda memory_type, memory_root, user_id: (None, None),
    )

    response = client.get(
        "/memory",
        params={"memoryType": "summary", "userId": "demo_user"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "memory": {"summary": "User is Andrew.", "updated_at": "2026-04-10T12:00:00Z"},
        "memoryPath": "memory_store/demo_user/summary.json",
        "factsEvents": None,
        "factsEventsPath": None,
        "promptPreview": "Long-Term Memory:\nUser is Andrew.",
    }


def test_delete_memory_endpoint_clears_snapshot(monkeypatch):
    monkeypatch.setattr(
        server,
        "delete_memory_snapshot",
        lambda memory_type, memory_root, user_id: (
            {"summary": "", "updated_at": None},
            server.Path("memory_store/demo_user/summary.json"),
        ),
    )
    monkeypatch.setattr(
        server,
        "build_system_prompt_preview",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        server,
        "read_facts_events_snapshot",
        lambda memory_type, memory_root, user_id: (
            [] if memory_type == "facts" else None,
            server.Path("memory_store/demo_user/facts_events.jsonl")
            if memory_type == "facts"
            else None,
        ),
    )

    response = client.delete(
        "/memory",
        params={"memoryType": "summary", "userId": "demo_user"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "memory": {"summary": "", "updated_at": None},
        "memoryPath": "memory_store/demo_user/summary.json",
        "factsEvents": None,
        "factsEventsPath": None,
        "promptPreview": None,
    }


def test_delete_memory_endpoint_clears_facts_snapshot_and_event_history(monkeypatch):
    monkeypatch.setattr(
        server,
        "delete_memory_snapshot",
        lambda memory_type, memory_root, user_id: (
            {"profile": {}},
            server.Path("memory_store/demo_user/facts.json"),
        ),
    )
    monkeypatch.setattr(
        server,
        "build_system_prompt_preview",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        server,
        "read_facts_events_snapshot",
        lambda memory_type, memory_root, user_id: (
            [],
            server.Path("memory_store/demo_user/facts_events.jsonl"),
        ),
    )

    response = client.delete(
        "/memory",
        params={"memoryType": "facts", "userId": "demo_user"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "memory": {"profile": {}},
        "memoryPath": "memory_store/demo_user/facts.json",
        "factsEvents": [],
        "factsEventsPath": "memory_store/demo_user/facts_events.jsonl",
        "promptPreview": None,
    }


def test_get_memory_endpoint_returns_facts_events_for_facts_mode(monkeypatch):
    monkeypatch.setattr(
        server,
        "read_memory_snapshot",
        lambda memory_type, memory_root, user_id: (
            {"profile": {"department": {"value": "Quality Assurance"}}},
            server.Path("memory_store/demo_user/facts.json"),
        ),
    )
    monkeypatch.setattr(
        server,
        "read_facts_events_snapshot",
        lambda memory_type, memory_root, user_id: (
            [
                {
                    "timestamp": "2026-04-10T12:34:56Z",
                    "category": "profile",
                    "key": "department",
                    "old_value": "Regulatory Affairs",
                    "new_value": "Quality Assurance",
                    "source": "Actually, I just transferred to Quality Assurance.",
                }
            ],
            server.Path("memory_store/demo_user/facts_events.jsonl"),
        ),
    )
    monkeypatch.setattr(
        server,
        "build_system_prompt_preview",
        lambda **kwargs: "Long-Term Memory:\nProfile:\n- department: Quality Assurance",
    )

    response = client.get(
        "/memory",
        params={"memoryType": "facts", "userId": "demo_user"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "memory": {"profile": {"department": {"value": "Quality Assurance"}}},
        "memoryPath": "memory_store/demo_user/facts.json",
        "factsEvents": [
            {
                "timestamp": "2026-04-10T12:34:56Z",
                "category": "profile",
                "key": "department",
                "old_value": "Regulatory Affairs",
                "new_value": "Quality Assurance",
                "source": "Actually, I just transferred to Quality Assurance.",
            }
        ],
        "factsEventsPath": "memory_store/demo_user/facts_events.jsonl",
        "promptPreview": "Long-Term Memory:\nProfile:\n- department: Quality Assurance",
    }


def test_finalize_session_endpoint_returns_refreshed_memory_and_events(monkeypatch):
    monkeypatch.setattr(
        server,
        "finalize_chat_session",
        lambda **kwargs: {
            "profile": {
                "name": {"value": "Dr. Sarah Doe"},
                "domain": {"value": "brain implant 510(k) submissions"},
            }
        },
    )
    monkeypatch.setattr(
        server,
        "read_facts_events_snapshot",
        lambda memory_type, memory_root, user_id: (
            [
                {
                    "timestamp": "2026-04-13T12:34:56Z",
                    "category": "profile",
                    "key": "name",
                    "old_value": "Dr. Sarah Chen",
                    "new_value": "Dr. Sarah Doe",
                    "source": "I got married and my last name is now Doe.",
                }
            ],
            server.Path("memory_store/demo_user/facts_events.jsonl"),
        ),
    )
    monkeypatch.setattr(
        server,
        "build_system_prompt_preview",
        lambda **kwargs: "Long-Term Memory:\nProfile:\n- name: Dr. Sarah Doe",
    )

    response = client.post(
        "/session/finalize",
        json={
            "messages": [{"role": "user", "content": "I got married and my last name is now Doe."}],
            "memoryType": "facts",
            "userId": "demo_user",
            "factsExtractor": "hybrid",
            "extractorModel": "anthropic:claude-haiku-4-5",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "memory": {
            "profile": {
                "name": {"value": "Dr. Sarah Doe"},
                "domain": {"value": "brain implant 510(k) submissions"},
            }
        },
        "memoryPath": "memory_store/demo_user/facts.json",
        "factsEvents": [
            {
                "timestamp": "2026-04-13T12:34:56Z",
                "category": "profile",
                "key": "name",
                "old_value": "Dr. Sarah Chen",
                "new_value": "Dr. Sarah Doe",
                "source": "I got married and my last name is now Doe.",
            }
        ],
        "factsEventsPath": "memory_store/demo_user/facts_events.jsonl",
        "promptPreview": "Long-Term Memory:\nProfile:\n- name: Dr. Sarah Doe",
    }
