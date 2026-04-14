from pathlib import Path

from agent.chat_service import (
    DEFAULT_CHAT_MODEL,
    delete_memory_snapshot,
    finalize_chat_session,
    run_chat_turn,
)
from agent.memory import facts_events_path, load_facts_memory, load_summary_memory
from agent.memory import (
    load_summary_facts_memory,
    summary_facts_events_path,
    summary_facts_path,
)


class FakeAgent:
    def __init__(self, recorder: dict):
        self.recorder = recorder

    def invoke(self, payload):
        self.recorder["messages"] = payload["messages"]
        return {
            "messages": [
                *payload["messages"],
                {"role": "assistant", "content": "Acknowledged."},
            ]
        }


def make_fake_agent_factory(recorder: dict):
    def factory(model_str, system_prompt=None):
        recorder["model_str"] = model_str
        recorder["system_prompt"] = system_prompt
        return FakeAgent(recorder)

    return factory


def test_run_chat_turn_persists_facts_after_each_turn(tmp_path: Path):
    recorder = {}

    result = run_chat_turn(
        messages=[{"role": "user", "content": "My name is Andrew. My preferred fruit is mango."}],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(recorder),
    )

    assert result["memory"]["profile"]["name"]["value"] == "Andrew"
    assert result["memory"]["preferences"]["preferred_fruit"]["value"] == "mango"
    assert "Long-Term Memory:" not in (result["system_prompt"] or "")
    assert (tmp_path / "demo_user" / "facts.json").exists()


def test_run_chat_turn_loads_saved_memory_into_system_prompt_on_restart(tmp_path: Path):
    first_recorder = {}
    run_chat_turn(
        messages=[{"role": "user", "content": "My name is Andrew. My preferred fruit is mango."}],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(first_recorder),
    )

    second_recorder = {}
    run_chat_turn(
        messages=[{"role": "user", "content": "What is my name and preferred fruit?"}],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(second_recorder),
    )

    assert "Andrew" in second_recorder["system_prompt"]
    assert "mango" in second_recorder["system_prompt"]


def test_run_chat_turn_none_mode_does_not_write_memory(tmp_path: Path):
    recorder = {}

    result = run_chat_turn(
        messages=[{"role": "user", "content": "My name is Andrew."}],
        memory_type="none",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(recorder),
    )

    assert result["memory"] is None
    assert result["system_prompt"] is None
    assert not (tmp_path / "demo_user").exists()
    assert recorder["model_str"] == DEFAULT_CHAT_MODEL


def test_run_chat_turn_persists_summary_after_each_turn(tmp_path: Path):
    recorder = {}

    result = run_chat_turn(
        messages=[{"role": "user", "content": "My name is Andrew. My preferred fruit is mango."}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(recorder),
    )

    assert "Andrew" in result["memory"]["summary"]
    assert "mango" in result["memory"]["summary"]
    assert (tmp_path / "demo_user" / "summary.json").exists()
    assert summary_facts_path(tmp_path, "demo_user").exists()


def test_run_chat_turn_summary_mode_keeps_latest_coherent_state_after_overwrite(
    tmp_path: Path,
):
    run_chat_turn(
        messages=[{"role": "user", "content": "My name is Andrew. My preferred fruit is mango."}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    result = run_chat_turn(
        messages=[{"role": "user", "content": "Actually, my preferred fruit is pear now."}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    assert result["memory"]["summary"] == "User is Andrew. Their preferred fruit is pear."
    stored_summary = load_summary_memory(tmp_path, "demo_user")
    assert stored_summary["summary"] == "User is Andrew. Their preferred fruit is pear."
    helper_facts = load_summary_facts_memory(tmp_path, "demo_user")
    assert helper_facts["preferences"]["preferred_fruit"]["value"] == "pear"


def test_run_chat_turn_returns_effective_system_prompt_preview(tmp_path: Path):
    recorder = {}

    run_chat_turn(
        messages=[{"role": "user", "content": "My name is Andrew. My preferred fruit is mango."}],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(recorder),
    )

    second = run_chat_turn(
        messages=[{"role": "user", "content": "What is my name and preferred fruit?"}],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    assert second["system_prompt"] is not None
    assert "Long-Term Memory:" in second["system_prompt"]
    assert "Use this long-term memory when relevant." in second["system_prompt"]


def test_run_chat_turn_summary_mode_does_not_inject_temporal_memory_for_normal_recall(
    tmp_path: Path,
):
    run_chat_turn(
        messages=[{"role": "user", "content": "My name is Andrew. My preferred fruit is mango."}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )
    run_chat_turn(
        messages=[{"role": "user", "content": "Actually, my preferred fruit is pear now."}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    recorder = {}
    run_chat_turn(
        messages=[{"role": "user", "content": "What is my name and preferred fruit now?"}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(recorder),
    )

    assert "Long-Term Memory:\nUser is Andrew. Their preferred fruit is pear." in recorder["system_prompt"]
    assert "Temporal Memory:" not in recorder["system_prompt"]
    assert "mango" not in recorder["system_prompt"]


def test_run_chat_turn_uses_department_label_for_restarted_facts_recall(tmp_path: Path):
    first_recorder = {}
    run_chat_turn(
        messages=[
            {"role": "user", "content": "I work in Regulatory Affairs."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "Actually, I just transferred to Quality Assurance."},
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(first_recorder),
    )

    second_recorder = {}
    run_chat_turn(
        messages=[{"role": "user", "content": "What department am I in now?"}],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(second_recorder),
    )

    assert "department: Quality Assurance" in second_recorder["system_prompt"]


def test_run_chat_turn_injects_temporal_memory_for_prior_state_questions(tmp_path: Path):
    recorder = {}
    messages = [{"role": "user", "content": "I work in Regulatory Affairs."}]
    first = run_chat_turn(
        messages=messages,
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(recorder),
    )

    second_messages = [
        *first["messages"],
        {"role": "user", "content": "Actually, I just transferred to Quality Assurance."},
    ]
    run_chat_turn(
        messages=second_messages,
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    third_recorder = {}
    run_chat_turn(
        messages=[{"role": "user", "content": "What was my job before?"}],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(third_recorder),
    )

    assert "Temporal Memory:" in third_recorder["system_prompt"]
    assert "department changed from Regulatory Affairs to Quality Assurance" in third_recorder["system_prompt"]


def test_run_chat_turn_summary_mode_injects_targeted_temporal_memory_for_prior_state_questions(
    tmp_path: Path,
):
    run_chat_turn(
        messages=[{"role": "user", "content": "My name is Andrew. My preferred fruit is mango."}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )
    run_chat_turn(
        messages=[{"role": "user", "content": "Actually, my preferred fruit is pear now."}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    recorder = {}
    recall = run_chat_turn(
        messages=[{"role": "user", "content": "What fruit did I prefer before my current one?"}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(recorder),
    )

    assert "Long-Term Memory:\nUser is Andrew. Their preferred fruit is pear." in recorder["system_prompt"]
    assert "Temporal Memory:" in recorder["system_prompt"]
    assert "preferred fruit changed from mango to pear" in recorder["system_prompt"]
    assert recall["memory"]["summary"] == "User is Andrew. Their preferred fruit is pear."
    assert "What fruit did I prefer before my current one?" not in load_summary_memory(
        tmp_path, "demo_user"
    )["summary"]


def test_run_chat_turn_summary_mode_does_not_create_temporal_event_log(tmp_path: Path):
    run_chat_turn(
        messages=[{"role": "user", "content": "My name is Andrew. My preferred fruit is mango."}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    assert not facts_events_path(tmp_path, "demo_user").exists()
    assert summary_facts_events_path(tmp_path, "demo_user").exists()


def test_run_chat_turn_updates_domain_and_name_for_common_update_phrases(tmp_path: Path):
    run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": (
                    "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a medical "
                    "device company, focusing on 510(k) submissions for cardiac devices."
                ),
            }
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    updated_domain = run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": "Now I focus on 510(k) submissions for brain implants.",
            },
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )
    updated_name = run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": "I got married and my last name is now Doe.",
            },
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    assert updated_domain["memory"]["profile"]["domain"]["value"] == "510(k) submissions for brain implants"
    assert updated_name["memory"]["profile"]["name"]["value"] == "Dr. Sarah Doe"


def test_run_chat_turn_returns_fact_events_for_common_update_phrases(tmp_path: Path):
    run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": (
                    "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a medical "
                    "device company, focusing on 510(k) submissions for cardiac devices."
                ),
            }
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": "Now I focus on 510(k) submissions for brain implants.",
            }
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )
    run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": "I got married and my last name is now Doe.",
            }
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    events = facts_events_path(tmp_path, "demo_user").read_text()
    assert "cardiac 510(k) submissions" in events
    assert "510(k) submissions for brain implants" in events
    assert "Dr. Sarah Chen" in events
    assert "Dr. Sarah Doe" in events


def test_run_chat_turn_does_not_corrupt_facts_memory_after_temporal_recall_question(
    tmp_path: Path,
):
    run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": (
                    "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a medical "
                    "device company, focusing on 510(k) submissions for cardiac devices."
                ),
            }
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )
    run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": "Now I focus on 510(k) submissions for brain implants.",
            },
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    recall_recorder = {}
    recall = run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": "What did I focus on before my current focus area?",
            }
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory(recall_recorder),
    )

    assert "Temporal Memory:" in recall_recorder["system_prompt"]
    assert "cardiac 510(k) submissions" in recall_recorder["system_prompt"]
    assert recall["memory"]["profile"]["domain"]["value"] == (
        "510(k) submissions for brain implants"
    )

    stored_facts = load_facts_memory(tmp_path, "demo_user")
    assert stored_facts["profile"]["domain"]["value"] == "510(k) submissions for brain implants"

    events = facts_events_path(tmp_path, "demo_user").read_text()
    assert "What did I focus on before my current focus area?" not in events


def test_run_chat_turn_does_not_corrupt_summary_memory_after_temporal_recall_question(
    tmp_path: Path,
):
    run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": (
                    "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a medical "
                    "device company, focusing on 510(k) submissions for cardiac devices."
                ),
            }
        ],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )
    run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": "Now I focus on 510(k) submissions for brain implants.",
            },
        ],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    recall = run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": "What did I focus on before my current focus area?",
            }
        ],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    assert "before my current focus area?" not in recall["memory"]["summary"]
    stored_summary = load_summary_memory(tmp_path, "demo_user")
    assert "before my current focus area?" not in stored_summary["summary"]


def test_finalize_chat_session_returns_llm_refined_facts(tmp_path: Path, monkeypatch):
    run_chat_turn(
        messages=[
            {
                "role": "user",
                "content": (
                    "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a medical "
                    "device company, focusing on 510(k) submissions for cardiac devices."
                ),
            }
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    monkeypatch.setattr(
        "agent.chat_service.llm_consolidate_facts_memory",
        lambda current_memory, transcript, extractor_model: {
            **current_memory,
            "profile": {
                **current_memory["profile"],
                "domain": {
                    "value": "brain implant 510(k) submissions",
                    "updated_at": "2026-04-13T12:34:56Z",
                    "source": "llm refinement",
                },
            },
        },
    )

    refined = finalize_chat_session(
        messages=[
            {
                "role": "user",
                "content": "Now I focus on 510(k) submissions for brain implants.",
            }
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        facts_extractor="hybrid",
    )

    assert refined["profile"]["domain"]["value"] == "brain implant 510(k) submissions"


def test_delete_memory_snapshot_for_facts_clears_event_log(tmp_path: Path):
    messages = [{"role": "user", "content": "I work in Regulatory Affairs."}]
    first = run_chat_turn(
        messages=messages,
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )
    run_chat_turn(
        messages=[
            *first["messages"],
            {"role": "user", "content": "Actually, I just transferred to Quality Assurance."},
        ],
        memory_type="facts",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    assert facts_events_path(tmp_path, "demo_user").exists()

    delete_memory_snapshot("facts", tmp_path, "demo_user")

    assert not facts_events_path(tmp_path, "demo_user").exists()


def test_delete_memory_snapshot_for_summary_clears_private_helper_state_and_events(
    tmp_path: Path,
):
    run_chat_turn(
        messages=[{"role": "user", "content": "My name is Andrew. My preferred fruit is mango."}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )
    run_chat_turn(
        messages=[{"role": "user", "content": "Actually, my preferred fruit is pear now."}],
        memory_type="summary",
        user_id="demo_user",
        memory_root=tmp_path,
        agent_factory=make_fake_agent_factory({}),
    )

    assert summary_facts_path(tmp_path, "demo_user").exists()
    assert summary_facts_events_path(tmp_path, "demo_user").exists()

    delete_memory_snapshot("summary", tmp_path, "demo_user")

    assert not summary_facts_path(tmp_path, "demo_user").exists()
    assert not summary_facts_events_path(tmp_path, "demo_user").exists()
