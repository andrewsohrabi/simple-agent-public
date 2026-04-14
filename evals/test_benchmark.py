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
