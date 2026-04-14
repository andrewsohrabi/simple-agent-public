from pathlib import Path

from agent.memory import (
    DEFAULT_FACTS_MEMORY,
    _canonicalize_llm_facts,
    append_facts_events,
    append_summary_facts_events,
    build_summary_memory_from_facts,
    consolidate_facts_memory,
    consolidate_summary_memory,
    facts_path,
    facts_events_path,
    load_facts_events,
    load_facts_memory,
    load_summary_facts_events,
    load_summary_facts_memory,
    load_summary_memory,
    render_facts_memory_block,
    render_summary_memory_block,
    render_temporal_facts_memory_block,
    save_facts_memory,
    save_summary_facts_memory,
    save_summary_memory,
    summary_facts_events_path,
    summary_facts_path,
    summary_path,
)


def test_path_resolution_by_user(tmp_path: Path):
    assert facts_path(tmp_path, "demo_user") == tmp_path / "demo_user" / "facts.json"
    assert facts_events_path(tmp_path, "demo_user") == tmp_path / "demo_user" / "facts_events.jsonl"
    assert summary_path(tmp_path, "demo_user") == tmp_path / "demo_user" / "summary.json"
    assert summary_facts_path(tmp_path, "demo_user") == (
        tmp_path / "demo_user__summary_internal" / "facts.json"
    )
    assert summary_facts_events_path(tmp_path, "demo_user") == (
        tmp_path / "demo_user__summary_internal" / "facts_events.jsonl"
    )


def test_load_facts_memory_returns_default_when_missing(tmp_path: Path):
    assert load_facts_memory(tmp_path, "demo_user") == DEFAULT_FACTS_MEMORY


def test_save_and_load_facts_memory_roundtrip(tmp_path: Path):
    facts_memory = {
        "profile": {
            "name": {
                "value": "Dr. Sarah Chen",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "User introduced themselves",
            }
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }

    save_facts_memory(tmp_path, "demo_user", facts_memory)

    assert load_facts_memory(tmp_path, "demo_user") == facts_memory


def test_load_summary_memory_returns_default_when_missing(tmp_path: Path):
    assert load_summary_memory(tmp_path, "demo_user") == {
        "summary": "",
        "updated_at": None,
    }


def test_save_and_load_summary_memory_roundtrip(tmp_path: Path):
    summary_memory = {
        "summary": "User prefers concise bullet-point answers.",
        "updated_at": "2026-04-10T12:34:56Z",
    }

    save_summary_memory(tmp_path, "demo_user", summary_memory)

    assert load_summary_memory(tmp_path, "demo_user") == summary_memory


def test_load_summary_facts_memory_returns_default_when_missing(tmp_path: Path):
    assert load_summary_facts_memory(tmp_path, "demo_user") == DEFAULT_FACTS_MEMORY


def test_save_and_load_summary_facts_memory_roundtrip(tmp_path: Path):
    facts_memory = {
        "profile": {},
        "preferences": {
            "preferred_fruit": {
                "value": "pear",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "fruit update",
            }
        },
        "constraints": {},
        "project_context": {},
    }

    save_summary_facts_memory(tmp_path, "demo_user", facts_memory)

    assert load_summary_facts_memory(tmp_path, "demo_user") == facts_memory


def test_render_facts_memory_block_is_deterministic():
    facts_memory = {
        "profile": {
            "name": {
                "value": "Dr. Sarah Chen",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "intro",
            },
            "department": {
                "value": "Regulatory Affairs",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "intro",
            },
            "role": {
                "value": "Regulatory Affairs",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "intro",
            },
        },
        "preferences": {
            "response_style": {
                "value": "concise bullet points",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "preference",
            }
        },
        "constraints": {},
        "project_context": {
            "current_project": {
                "value": "new catheter 510(k)",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "project",
            }
        },
    }

    assert render_facts_memory_block(facts_memory) == (
        "Long-Term Memory:\n"
        "Profile:\n"
        "- name: Dr. Sarah Chen\n"
        "- department: Regulatory Affairs\n\n"
        "Preferences:\n"
        "- response_style: concise bullet points\n\n"
        "Project Context:\n"
        "- current_project: new catheter 510(k)"
    )


def test_render_summary_memory_block_is_deterministic():
    summary_memory = {
        "summary": (
            "User is Dr. Sarah Chen in Regulatory Affairs focused on cardiac 510(k) "
            "submissions."
        ),
        "updated_at": "2026-04-10T12:34:56Z",
    }

    assert render_summary_memory_block(summary_memory) == (
        "Long-Term Memory:\n"
        "User is Dr. Sarah Chen in Regulatory Affairs focused on cardiac 510(k) submissions."
    )


def test_build_summary_memory_from_facts_uses_latest_coherent_state():
    facts_memory = {
        "profile": {
            "name": {
                "value": "Andrew",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "intro",
            }
        },
        "preferences": {
            "preferred_fruit": {
                "value": "pear",
                "updated_at": "2026-04-10T12:35:56Z",
                "source": "fruit update",
            }
        },
        "constraints": {},
        "project_context": {},
    }

    summary = build_summary_memory_from_facts(
        facts_memory,
        previous_summary={
            "summary": "User is Andrew. Their preferred fruit is mango.",
            "updated_at": "2026-04-10T12:00:00Z",
        },
        now="2026-04-10T12:36:00Z",
    )

    assert summary["summary"] == "User is Andrew. Their preferred fruit is pear."
    assert summary["updated_at"] == "2026-04-10T12:36:00Z"


def test_append_facts_events_records_only_canonical_fact_changes(tmp_path: Path):
    previous = {
        "profile": {
            "department": {
                "value": "Regulatory Affairs",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "old session",
            },
            "role": {
                "value": "Regulatory Affairs",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "old session",
            },
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }
    updated = {
        "profile": {
            "department": {
                "value": "Quality Assurance",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "Actually, I just transferred to Quality Assurance.",
            },
            "role": {
                "value": "Quality Assurance",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "Actually, I just transferred to Quality Assurance.",
            },
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }

    events = append_facts_events(tmp_path, "demo_user", previous, updated)

    assert events == [
        {
            "timestamp": "2026-04-10T12:34:56Z",
            "category": "profile",
            "key": "department",
            "old_value": "Regulatory Affairs",
            "new_value": "Quality Assurance",
            "source": "Actually, I just transferred to Quality Assurance.",
        }
    ]
    assert load_facts_events(tmp_path, "demo_user") == events


def test_render_temporal_facts_memory_block_filters_to_relevant_history():
    events = [
        {
            "timestamp": "2026-04-10T12:34:56Z",
            "category": "profile",
            "key": "department",
            "old_value": "Regulatory Affairs",
            "new_value": "Quality Assurance",
            "source": "transfer",
        },
        {
            "timestamp": "2026-04-10T12:35:30Z",
            "category": "preferences",
            "key": "preferred_fruit",
            "old_value": "mango",
            "new_value": "pear",
            "source": "fruit update",
        },
    ]

    block = render_temporal_facts_memory_block(events, "What was my job before?")

    assert "Temporal Memory:" in block
    assert "department changed from Regulatory Affairs to Quality Assurance" in block
    assert "preferred fruit" not in block


def test_render_temporal_facts_memory_block_filters_to_domain_history_for_focus_queries():
    events = [
        {
            "timestamp": "2026-04-10T12:34:56Z",
            "category": "profile",
            "key": "domain",
            "old_value": "cardiac 510(k) submissions",
            "new_value": "510(k) submissions for brain implants",
            "source": "focus update",
        },
        {
            "timestamp": "2026-04-10T12:35:30Z",
            "category": "preferences",
            "key": "preferred_fruit",
            "old_value": "mango",
            "new_value": "pear",
            "source": "fruit update",
        },
    ]

    block = render_temporal_facts_memory_block(
        events,
        "What did I focus on before my current focus area?",
    )

    assert "domain changed from cardiac 510(k) submissions to 510(k) submissions for brain implants" in block
    assert "preferred fruit" not in block


def test_append_summary_facts_events_records_preference_overwrite(tmp_path: Path):
    previous = {
        "profile": {},
        "preferences": {
            "preferred_fruit": {
                "value": "mango",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "old session",
            }
        },
        "constraints": {},
        "project_context": {},
    }
    updated = {
        "profile": {},
        "preferences": {
            "preferred_fruit": {
                "value": "pear",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "Actually, my preferred fruit is pear now.",
            }
        },
        "constraints": {},
        "project_context": {},
    }

    events = append_summary_facts_events(tmp_path, "demo_user", previous, updated)

    assert events == [
        {
            "timestamp": "2026-04-10T12:34:56Z",
            "category": "preferences",
            "key": "preferred_fruit",
            "old_value": "mango",
            "new_value": "pear",
            "source": "Actually, my preferred fruit is pear now.",
        }
    ]
    assert load_summary_facts_events(tmp_path, "demo_user") == events


def test_consolidate_facts_memory_extracts_structured_facts():
    transcript = [
        {
            "role": "user",
            "content": (
                "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a medical "
                "device company, focusing on 510(k) submissions for cardiac devices. "
                "I prefer concise bullet-point answers."
            ),
        },
        {"role": "assistant", "content": "Thanks, Dr. Chen."},
    ]

    updated = consolidate_facts_memory(
        DEFAULT_FACTS_MEMORY,
        transcript,
        now="2026-04-10T12:34:56Z",
    )

    assert updated["profile"]["name"]["value"] == "Dr. Sarah Chen"
    assert updated["profile"]["department"]["value"] == "Regulatory Affairs"
    assert updated["profile"]["role"]["value"] == "Regulatory Affairs"
    assert updated["profile"]["domain"]["value"] == "cardiac 510(k) submissions"
    assert updated["preferences"]["response_style"]["value"] == "concise bullet-point answers"


def test_consolidate_facts_memory_updates_response_style_for_answer_format_override():
    current = consolidate_facts_memory(
        DEFAULT_FACTS_MEMORY,
        [
            {
                "role": "user",
                "content": "Going forward, always give me three-line haiku answers.",
            }
        ],
        now="2026-04-10T12:34:56Z",
    )

    updated = consolidate_facts_memory(
        current,
        [
            {
                "role": "user",
                "content": (
                    "Update that: start every answer with an ALL-CAPS summary line, "
                    "then give me concise bullet-point answers. No haikus."
                ),
            }
        ],
        now="2026-04-10T12:35:56Z",
    )

    assert updated["preferences"]["response_style"]["value"] == (
        "start every answer with an ALL-CAPS summary line, then give me concise "
        "bullet-point answers"
    )


def test_consolidate_facts_memory_extracts_plain_name_and_preferred_fruit():
    transcript = [
        {
            "role": "user",
            "content": "My name is Andrew. My preferred fruit is mango.",
        }
    ]

    updated = consolidate_facts_memory(
        DEFAULT_FACTS_MEMORY,
        transcript,
        now="2026-04-10T12:34:56Z",
    )

    assert updated["profile"]["name"]["value"] == "Andrew"
    assert updated["preferences"]["preferred_fruit"]["value"] == "mango"


def test_consolidate_facts_memory_extracts_preference_comparison():
    transcript = [
        {
            "role": "user",
            "content": "I prefer chocolate over peanut butter.",
        }
    ]

    updated = consolidate_facts_memory(
        DEFAULT_FACTS_MEMORY,
        transcript,
        now="2026-04-10T12:34:56Z",
    )

    assert updated["preferences"]["preference_comparison"]["value"] == (
        "chocolate over peanut butter"
    )


def test_consolidate_facts_memory_does_not_map_generic_food_preference_to_response_style():
    transcript = [
        {
            "role": "user",
            "content": "I prefer mangoes.",
        }
    ]

    updated = consolidate_facts_memory(
        DEFAULT_FACTS_MEMORY,
        transcript,
        now="2026-04-10T12:34:56Z",
    )

    assert "response_style" not in updated["preferences"]


def test_consolidate_facts_memory_uses_latest_value_for_contradiction():
    existing = {
        "profile": {
            "department": {
                "value": "Regulatory Affairs",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "old session",
            },
            "role": {
                "value": "Regulatory Affairs",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "old session",
            }
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }
    transcript = [
        {
            "role": "user",
            "content": "Actually, I just transferred to Quality Assurance.",
        }
    ]

    updated = consolidate_facts_memory(existing, transcript, now="2026-04-10T12:34:56Z")

    assert updated["profile"]["department"]["value"] == "Quality Assurance"
    assert updated["profile"]["department"]["updated_at"] == "2026-04-10T12:34:56Z"
    assert updated["profile"]["role"]["value"] == "Quality Assurance"
    assert updated["profile"]["role"]["updated_at"] == "2026-04-10T12:34:56Z"


def test_consolidate_facts_memory_updates_domain_from_focus_phrasing():
    existing = {
        "profile": {
            "domain": {
                "value": "cardiac 510(k) submissions",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "old session",
            }
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }
    transcript = [
        {
            "role": "user",
            "content": "Now I focus on 510(k) submissions for brain implants.",
        }
    ]

    updated = consolidate_facts_memory(existing, transcript, now="2026-04-10T12:34:56Z")

    assert updated["profile"]["domain"]["value"] == "510(k) submissions for brain implants"
    assert updated["profile"]["domain"]["updated_at"] == "2026-04-10T12:34:56Z"


def test_consolidate_facts_memory_updates_existing_name_from_last_name_change():
    existing = {
        "profile": {
            "name": {
                "value": "Dr. Sarah Chen",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "old session",
            }
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }
    transcript = [
        {
            "role": "user",
            "content": "I got married and my last name is now Doe.",
        }
    ]

    updated = consolidate_facts_memory(existing, transcript, now="2026-04-10T12:34:56Z")

    assert updated["profile"]["name"]["value"] == "Dr. Sarah Doe"
    assert updated["profile"]["name"]["updated_at"] == "2026-04-10T12:34:56Z"


def test_consolidate_facts_memory_ignores_question_only_focus_recall_prompts():
    existing = {
        "profile": {
            "domain": {
                "value": "510(k) submissions for brain implants",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "old session",
            }
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }
    transcript = [
        {
            "role": "user",
            "content": "What did I focus on before my current focus area?",
        }
    ]

    updated = consolidate_facts_memory(existing, transcript, now="2026-04-10T12:34:56Z")

    assert updated["profile"]["domain"]["value"] == "510(k) submissions for brain implants"
    assert updated["profile"]["domain"]["updated_at"] == "2026-04-10T12:00:00Z"


def test_consolidate_facts_memory_keeps_declarative_clauses_and_ignores_questions():
    transcript = [
        {
            "role": "user",
            "content": "My name is Andrew. What did I focus on before my current focus area?",
        }
    ]

    updated = consolidate_facts_memory(
        DEFAULT_FACTS_MEMORY,
        transcript,
        now="2026-04-10T12:34:56Z",
    )

    assert updated["profile"]["name"]["value"] == "Andrew"
    assert "domain" not in updated["profile"]


def test_consolidate_summary_memory_ignores_question_only_focus_recall_prompts():
    current_summary = {
        "summary": "User focused on 510(k) submissions for brain implants.",
        "updated_at": "2026-04-10T12:00:00Z",
    }
    transcript = [
        {
            "role": "user",
            "content": "What did I focus on before my current focus area?",
        }
    ]

    updated = consolidate_summary_memory(
        current_summary,
        transcript,
        now="2026-04-10T12:34:56Z",
    )

    assert updated["summary"] == "User focused on 510(k) submissions for brain implants."


def test_canonicalize_llm_facts_ignores_noncanonical_temporal_keys():
    current_memory = {
        "profile": {
            "domain": {
                "value": "510(k) submissions for brain implants",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "old session",
            }
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }
    raw = {
        "profile": {
            "domain": "510(k) submissions for brain implants",
            "prior_domain": "cardiac 510(k) submissions",
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }

    updated = _canonicalize_llm_facts(
        raw,
        current_memory,
        [{"role": "user", "content": "What did I focus on before my current focus area?"}],
        "2026-04-10T12:34:56Z",
    )

    assert updated["profile"]["domain"]["value"] == "510(k) submissions for brain implants"
    assert "prior_domain" not in updated["profile"]


def test_canonicalize_llm_facts_preserves_metadata_for_unchanged_values():
    current_memory = {
        "profile": {
            "domain": {
                "value": "510(k) submissions for brain implants",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "Now I focus on 510(k) submissions for brain implants.",
            }
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }
    raw = {
        "profile": {"domain": "510(k) submissions for brain implants"},
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }

    updated = _canonicalize_llm_facts(
        raw,
        current_memory,
        [{"role": "user", "content": "What did I focus on before my current focus area?"}],
        "2026-04-10T12:34:56Z",
    )

    assert updated["profile"]["domain"] == current_memory["profile"]["domain"]


def test_append_facts_events_records_domain_and_name_updates(tmp_path: Path):
    previous = {
        "profile": {
            "name": {
                "value": "Dr. Sarah Chen",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "old session",
            },
            "domain": {
                "value": "cardiac 510(k) submissions",
                "updated_at": "2026-04-10T12:00:00Z",
                "source": "old session",
            },
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }
    updated = {
        "profile": {
            "name": {
                "value": "Dr. Sarah Doe",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "I got married and my last name is now Doe.",
            },
            "domain": {
                "value": "510(k) submissions for brain implants",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "Now I focus on 510(k) submissions for brain implants.",
            },
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }

    events = append_facts_events(tmp_path, "demo_user", previous, updated)

    assert events == [
        {
            "timestamp": "2026-04-10T12:34:56Z",
            "category": "profile",
            "key": "domain",
            "old_value": "cardiac 510(k) submissions",
            "new_value": "510(k) submissions for brain implants",
            "source": "Now I focus on 510(k) submissions for brain implants.",
        },
        {
            "timestamp": "2026-04-10T12:34:56Z",
            "category": "profile",
            "key": "name",
            "old_value": "Dr. Sarah Chen",
            "new_value": "Dr. Sarah Doe",
            "source": "I got married and my last name is now Doe.",
        },
    ]


def test_render_facts_memory_block_prefers_department_label_when_role_matches():
    facts_memory = {
        "profile": {
            "department": {
                "value": "Quality Assurance",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "update",
            },
            "role": {
                "value": "Quality Assurance",
                "updated_at": "2026-04-10T12:34:56Z",
                "source": "update",
            },
        },
        "preferences": {},
        "constraints": {},
        "project_context": {},
    }

    assert render_facts_memory_block(facts_memory) == (
        "Long-Term Memory:\n"
        "Profile:\n"
        "- department: Quality Assurance"
    )


def test_consolidate_facts_memory_ignores_irrelevant_transient_content():
    transcript = [
        {"role": "user", "content": "Hello! What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]

    updated = consolidate_facts_memory(
        DEFAULT_FACTS_MEMORY,
        transcript,
        now="2026-04-10T12:34:56Z",
    )

    assert updated == DEFAULT_FACTS_MEMORY


def test_consolidate_summary_memory_builds_bounded_summary():
    transcript = [
        {
            "role": "user",
            "content": (
                "I'm Dr. Sarah Chen. I work in Regulatory Affairs focusing on "
                "cardiac 510(k) submissions. I prefer concise bullet-point answers."
            ),
        }
    ]

    updated = consolidate_summary_memory(
        {"summary": "", "updated_at": None},
        transcript,
        now="2026-04-10T12:34:56Z",
        max_chars=220,
    )

    assert "Dr. Sarah Chen" in updated["summary"]
    assert "Regulatory Affairs" in updated["summary"]
    assert "Regulatory Affairs" in updated["summary"]
    assert "concise bullet-point answers" in updated["summary"]
    assert len(updated["summary"]) <= 220
    assert updated["updated_at"] == "2026-04-10T12:34:56Z"


def test_consolidate_summary_memory_builds_summary_for_plain_name_and_preferred_fruit():
    transcript = [
        {
            "role": "user",
            "content": "My name is Andrew. My preferred fruit is mango.",
        }
    ]

    updated = consolidate_summary_memory(
        {"summary": "", "updated_at": None},
        transcript,
        now="2026-04-10T12:34:56Z",
        max_chars=220,
    )

    assert "Andrew" in updated["summary"]
    assert "mango" in updated["summary"]
