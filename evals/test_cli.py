from agent.cli import build_parser, compose_system_prompt


def test_parser_defaults_to_no_memory_and_demo_user():
    args = build_parser().parse_args([])

    assert args.memory_type == "none"
    assert args.user_id == "demo_user"
    assert args.facts_extractor == "hybrid"


def test_parser_accepts_memory_flags():
    args = build_parser().parse_args(
        [
            "--memory-type",
            "facts",
            "--user-id",
            "sarah",
            "--memory-root",
            "tmp-memory",
            "--facts-extractor",
            "deterministic",
            "--extractor-model",
            "anthropic:test-extractor",
        ]
    )

    assert args.memory_type == "facts"
    assert args.user_id == "sarah"
    assert args.memory_root == "tmp-memory"
    assert args.facts_extractor == "deterministic"
    assert args.extractor_model == "anthropic:test-extractor"


def test_compose_system_prompt_preserves_user_prompt_and_memory_block():
    assert compose_system_prompt("Base prompt", "Long-Term Memory:\n- name: Sarah") == (
        "Base prompt\n\n"
        "Long-Term Memory:\n"
        "- name: Sarah\n\n"
        "Use this long-term memory when relevant. Follow durable preferences and "
        "constraints unless the current user request explicitly overrides them."
    )


def test_compose_system_prompt_handles_missing_base_prompt():
    assert compose_system_prompt(None, "Long-Term Memory:\n- name: Sarah") == (
        "Long-Term Memory:\n"
        "- name: Sarah\n\n"
        "Use this long-term memory when relevant. Follow durable preferences and "
        "constraints unless the current user request explicitly overrides them."
    )
