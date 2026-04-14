from pathlib import Path

from harness import (
    SCENARIOS,
    audit_artifacts,
    artifact_paths,
    build_cli_command,
    _format_run_header,
    _format_terminal_summary,
    score_forgetful_output,
    score_output,
    score_style_output,
)


def test_harness_defines_core_scenarios():
    assert [scenario["name"] for scenario in SCENARIOS] == [
        "identity_recall",
        "preference_application",
        "project_context_recall",
        "contradiction_update",
        "personal_preference_recall",
    ]


def test_harness_scenarios_define_current_and_prior_state_checks():
    expectations = {
        "identity_recall": {
            "session_1": [
                (
                    "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a "
                    "medical device company, focusing on 510(k) submissions for "
                    "cardiac devices."
                ),
                "I got married and my last name is now Doe.",
                "Now I focus on 510(k) submissions for brain implants.",
            ],
            "session_2_prompts": [
                "What do you remember about me and my work now?",
                "What did I focus on before my current focus area?",
            ],
            "checks": [
                {"scorer": "terms", "expected_terms": ["Sarah", "Doe", "Regulatory", "brain implant"]},
                {"scorer": "terms", "expected_terms": ["cardiac", "510(k)"]},
            ],
        },
        "preference_application": {
            "session_1": [
                "Going forward, always give me three-line haiku answers.",
                "Update that: start every answer with an ALL-CAPS summary line, then give me concise bullet-point answers. No haikus.",
            ],
            "session_2_prompts": [
                "Explain the key considerations for predicate device selection in a 510(k).",
                "What response style did I ask for before my current one?",
            ],
            "facts_checks": [
                {"scorer": "style"},
                {"scorer": "terms", "expected_terms": ["haiku"]},
            ],
            "summary_checks": [
                {"scorer": "style"},
                {"scorer": "terms", "expected_terms": ["haiku"]},
            ],
        },
        "project_context_recall": {
            "session_1": [
                "I'm working on a 510(k) for a new catheter. The main challenge is choosing between two predicate devices.",
                "Now I'm working on a 510(k) for a brain implant. The main challenge is building the clinical evidence plan.",
            ],
            "session_2_prompts": [
                "Can you help me think through next steps for my current project?",
                "What project was I working on before my current one?",
            ],
            "checks": [
                {"scorer": "terms", "expected_terms": ["brain implant", "clinical evidence"]},
                {"scorer": "terms", "expected_terms": ["catheter", "predicate"]},
            ],
        },
        "contradiction_update": {
            "session_1": [
                "I work in Regulatory Affairs.",
                "Actually, I just transferred to Quality Assurance.",
            ],
            "session_2_prompts": [
                "What department am I in now?",
                "What was my job before?",
            ],
            "checks": [
                {"scorer": "terms", "expected_terms": ["Quality Assurance"]},
                {"scorer": "terms", "expected_terms": ["Regulatory Affairs"]},
            ],
        },
        "personal_preference_recall": {
            "session_1": [
                "My name is Andrew. My preferred fruit is mango.",
                "Actually, my preferred fruit is pear now.",
            ],
            "session_2_prompts": [
                "What is my name and preferred fruit now?",
                "What fruit did I prefer before my current one?",
            ],
            "checks": [
                {"scorer": "terms", "expected_terms": ["Andrew", "pear"]},
                {"scorer": "terms", "expected_terms": ["mango"]},
            ],
        },
    }

    for scenario in SCENARIOS:
        expected = expectations[scenario["name"]]
        assert scenario["sessions"][0] == expected["session_1"]
        assert scenario["sessions"][1] == expected["session_2_prompts"]
        assert scenario["expectations"]["none"]["checks"] == [
            {"scorer": "forgetful"},
            {"scorer": "forgetful"},
        ]
        if "checks" in expected:
            assert scenario["expectations"]["facts"]["checks"] == expected["checks"]
            assert scenario["expectations"]["summary"]["checks"] == expected["checks"]
        else:
            assert scenario["expectations"]["facts"]["checks"] == expected["facts_checks"]
            assert scenario["expectations"]["summary"]["checks"] == expected["summary_checks"]


def test_build_cli_command_uses_strategy_user_and_memory_root(monkeypatch):
    monkeypatch.setattr("harness.sys.executable", "python")
    command = build_cli_command(
        memory_type="facts",
        user_id="demo_identity_facts",
        memory_root=Path("memory_store"),
        model="anthropic:test-model",
        facts_extractor="hybrid",
        extractor_model="anthropic:test-extractor",
    )

    assert command == [
        "python",
        "-m",
        "agent.cli",
        "--model",
        "anthropic:test-model",
        "--memory-type",
        "facts",
        "--user-id",
        "demo_identity_facts",
        "--memory-root",
        "memory_store",
        "--facts-extractor",
        "hybrid",
        "--extractor-model",
        "anthropic:test-extractor",
    ]


def test_build_cli_command_skips_facts_flags_for_non_facts_modes(monkeypatch):
    monkeypatch.setattr("harness.sys.executable", "python")
    command = build_cli_command(
        memory_type="summary",
        user_id="demo_summary",
        memory_root=Path("memory_store"),
        model="anthropic:test-model",
        facts_extractor="hybrid",
        extractor_model="anthropic:test-extractor",
    )

    assert "--facts-extractor" not in command
    assert "--extractor-model" not in command


def test_artifact_paths_are_strategy_and_scenario_scoped(tmp_path: Path):
    paths = artifact_paths(tmp_path, "facts", "identity_recall")

    assert paths["dir"] == tmp_path / "facts" / "identity_recall"
    assert paths["session_1"] == tmp_path / "facts" / "identity_recall" / "session_1.md"
    assert paths["memory_after_session_1"] == (
        tmp_path / "facts" / "identity_recall" / "memory_after_session_1.json"
    )
    assert paths["session_2"] == tmp_path / "facts" / "identity_recall" / "session_2.md"


def test_score_output_pass_partial_and_fail():
    assert score_output("Dr. Sarah Chen works in Regulatory Affairs", ["Sarah", "Regulatory"]) == "pass"
    assert score_output("Sarah works in medical devices", ["Sarah", "Regulatory"]) == "partial"
    assert score_output("I do not know yet", ["Sarah", "Regulatory"]) == "fail"


def test_score_style_output_checks_caps_bullets_and_non_haiku_shape():
    concise_bullets = "SUMMARY LINE\n- First point\n- Second point\n- Third point"
    verbose_bullets = "SUMMARY LINE\n" + "\n".join(
        f"- point {index} " + ("word " * 80) for index in range(4))
    prose = "This is a normal paragraph without a durable bullet-point format."
    haiku = "quiet predicate path\nregulators compare devices\nclearance drifts downstream"

    assert score_style_output(concise_bullets) == "pass"
    assert score_style_output(verbose_bullets) == "partial"
    assert score_style_output(prose) == "fail"
    assert score_style_output(haiku) == "fail"


def test_score_forgetful_output_accepts_explicit_not_knowing():
    assert score_forgetful_output("I don't have any information about that yet.") == "pass"
    assert score_forgetful_output("You are Andrew and your preferred fruit is pear.") == "fail"


def test_terminal_output_includes_runtime_configuration():
    results = {
        "config": {
            "model": "anthropic:test-model",
            "facts_extractor": "hybrid",
            "extractor_model": "anthropic:test-extractor",
            "jobs": 15,
            "strategies": ["none", "facts", "summary"],
            "scenarios": ["identity_recall"],
        },
        "scenarios": {
            "identity_recall": {
                "none": {"outcome": "fail"},
                "facts": {"outcome": "pass"},
                "summary": {"outcome": "partial"},
            }
        },
    }

    summary = _format_terminal_summary(results)
    header = _format_run_header(results["config"])

    assert "model=anthropic:test-model" in summary
    assert "facts_extractor=hybrid" in summary
    assert "extractor_model=anthropic:test-extractor" in summary
    assert "jobs=15" in summary
    assert "strategies=none, facts, summary" in header
    assert "scenarios=identity_recall" in header


def test_audit_artifacts_flags_summary_append_drift(tmp_path: Path):
    user_id = "demo_user"
    user_root = tmp_path / user_id
    user_root.mkdir(parents=True)
    (user_root / "summary.json").write_text(
        '{\n'
        '  "summary": "User is Andrew. Their preferred fruit is mango. Their preferred fruit is pear.",\n'
        '  "updated_at": "2026-04-13T22:57:29Z"\n'
        '}\n'
    )
    internal_root = tmp_path / f"{user_id}__summary_internal"
    internal_root.mkdir(parents=True)
    (internal_root / "facts_events.jsonl").write_text(
        '{"key":"preferred_fruit","old_value":"mango","new_value":"pear"}\n'
    )

    audit = audit_artifacts(
        strategy="summary",
        scenario=next(item for item in SCENARIOS if item["name"] == "personal_preference_recall"),
        memory_root=tmp_path,
        user_id=user_id,
    )

    assert audit["outcome"] == "fail"
    assert any(check["outcome"] == "fail" for check in audit["checks"])
