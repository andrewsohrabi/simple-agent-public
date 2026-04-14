import benchmark_facts_extractor as benchmark
from benchmark_facts_extractor import CASES, compare_flat_facts


def test_benchmark_cases_cover_manual_and_take_home_flows():
    assert [case["name"] for case in CASES] == [
        "plain_name",
        "name_and_preferred_fruit",
        "role_department",
        "response_style",
        "project_context",
        "contradiction_update",
        "question_only_no_memory",
        "mixed_durable_and_transient",
        "generic_preference_not_response_style",
    ]


def test_compare_flat_facts_scores_precision_recall_and_exact():
    expected = {
        "profile": {"name": "Andrew"},
        "preferences": {"preferred_fruit": "mango"},
        "constraints": {},
        "project_context": {},
    }
    actual = {
        "profile": {"name": "Andrew"},
        "preferences": {"preferred_fruit": "mango", "response_style": "concise"},
        "constraints": {},
        "project_context": {},
    }

    score = compare_flat_facts(expected, actual)

    assert score["true_positives"] == 2
    assert score["false_positives"] == 1
    assert score["false_negatives"] == 0
    assert score["exact"] is False


def test_run_benchmark_skips_llm_when_auth_is_missing(tmp_path, monkeypatch):
    dotenv_calls = []

    monkeypatch.setattr(benchmark, "load_dotenv", lambda: dotenv_calls.append("called"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        benchmark,
        "llm_consolidate_facts_memory",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM extractor should be skipped")),
    )

    payload = benchmark.run_benchmark("anthropic:test-extractor", tmp_path)

    assert dotenv_calls == ["called"]
    assert payload["summary"]["deterministic"]["status"] == "scored"
    assert payload["summary"]["llm"] == {
        "status": "skipped",
        "reason": "Skipping llm benchmark: extractor model anthropic:test-extractor requires ANTHROPIC_API_KEY.",
    }
    assert payload["cases"]["llm"][0]["status"] == "skipped"
    assert payload["cases"]["llm"][0]["reason"] == (
        "Skipping llm benchmark: extractor model anthropic:test-extractor requires ANTHROPIC_API_KEY."
    )


def test_run_benchmark_scores_llm_when_auth_is_present(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        benchmark,
        "CASES",
        [
            {
                "name": "plain_name",
                "transcript": [{"role": "user", "content": "My name is Andrew."}],
                "expected": {
                    "profile": {"name": "Andrew"},
                    "preferences": {},
                    "constraints": {},
                    "project_context": {},
                },
            }
        ],
    )
    monkeypatch.setattr(
        benchmark,
        "llm_consolidate_facts_memory",
        lambda *args, **kwargs: {
            "profile": {
                "name": {
                    "value": "Andrew",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "llm",
                }
            },
            "preferences": {},
            "constraints": {},
            "project_context": {},
        },
    )

    payload = benchmark.run_benchmark("anthropic:test-extractor", tmp_path)

    assert payload["summary"]["llm"] == {
        "status": "scored",
        "cases": 1,
        "exact": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert payload["cases"]["llm"][0]["score"]["exact"] is True


def test_format_results_markdown_renders_skipped_extractors_without_zero_scores():
    payload = {
        "summary": {
            "deterministic": {
                "status": "scored",
                "cases": 1,
                "exact": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
            "llm": {
                "status": "skipped",
                "reason": "Skipping llm benchmark: extractor model anthropic:test-extractor requires ANTHROPIC_API_KEY.",
            },
        },
        "cases": {
            "deterministic": [
                {
                    "name": "plain_name",
                    "score": {"exact": True, "precision": 1.0, "recall": 1.0, "f1": 1.0},
                }
            ],
            "llm": [
                {
                    "name": "plain_name",
                    "status": "skipped",
                    "reason": "Skipping llm benchmark: extractor model anthropic:test-extractor requires ANTHROPIC_API_KEY.",
                }
            ],
        },
    }

    markdown = benchmark.format_results_markdown(payload)

    assert "| llm | skipped | - | - | - | - | - | Skipping llm benchmark: extractor model anthropic:test-extractor requires ANTHROPIC_API_KEY. |" in markdown
    assert "- `plain_name`: skipped (Skipping llm benchmark: extractor model anthropic:test-extractor requires ANTHROPIC_API_KEY.)" in markdown
