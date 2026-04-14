from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.chat_service import DEFAULT_EXTRACTOR_MODEL
from agent.memory import (
    DEFAULT_FACTS_MEMORY,
    consolidate_facts_memory,
    flatten_facts_memory,
    llm_consolidate_facts_memory,
)


CASES: list[dict[str, Any]] = [
    {
        "name": "plain_name",
        "transcript": [{"role": "user", "content": "My name is Andrew."}],
        "expected": {"profile": {"name": "Andrew"}, "preferences": {}, "constraints": {}, "project_context": {}},
    },
    {
        "name": "name_and_preferred_fruit",
        "transcript": [{"role": "user", "content": "My name is Andrew. My preferred fruit is mango."}],
        "expected": {
            "profile": {"name": "Andrew"},
            "preferences": {"preferred_fruit": "mango"},
            "constraints": {},
            "project_context": {},
        },
    },
    {
        "name": "role_department",
        "transcript": [{"role": "user", "content": "I work in Quality Assurance."}],
        "expected": {
            "profile": {
                "department": "Quality Assurance",
                "role": "Quality Assurance",
            },
            "preferences": {},
            "constraints": {},
            "project_context": {},
        },
    },
    {
        "name": "response_style",
        "transcript": [{"role": "user", "content": "Going forward, always give me concise bullet-point answers. No long paragraphs."}],
        "expected": {
            "profile": {},
            "preferences": {"response_style": "concise bullet-point answers"},
            "constraints": {"avoid_long_paragraphs": "no long paragraphs"},
            "project_context": {},
        },
    },
    {
        "name": "project_context",
        "transcript": [{"role": "user", "content": "I'm working on a 510(k) for a new catheter. The main challenge is choosing between two predicate devices."}],
        "expected": {
            "profile": {},
            "preferences": {},
            "constraints": {},
            "project_context": {
                "current_project": "new catheter 510(k)",
                "key_challenge": "choosing between two predicate devices",
            },
        },
    },
    {
        "name": "contradiction_update",
        "transcript": [
            {"role": "user", "content": "I work in Regulatory Affairs."},
            {"role": "user", "content": "Actually, I just transferred to Quality Assurance."},
        ],
        "expected": {
            "profile": {
                "department": "Quality Assurance",
                "role": "Quality Assurance",
            },
            "preferences": {},
            "constraints": {},
            "project_context": {},
        },
    },
    {
        "name": "question_only_no_memory",
        "transcript": [{"role": "user", "content": "What is my name and preferred fruit?"}],
        "expected": {"profile": {}, "preferences": {}, "constraints": {}, "project_context": {}},
    },
    {
        "name": "mixed_durable_and_transient",
        "transcript": [
            {"role": "user", "content": "My name is Andrew. What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "My preferred fruit is mango."},
        ],
        "expected": {
            "profile": {"name": "Andrew"},
            "preferences": {"preferred_fruit": "mango"},
            "constraints": {},
            "project_context": {},
        },
    },
    {
        "name": "generic_preference_not_response_style",
        "transcript": [{"role": "user", "content": "I prefer mangoes."}],
        "expected": {"profile": {}, "preferences": {}, "constraints": {}, "project_context": {}},
    },
]


def compare_flat_facts(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    expected_pairs = _flatten_pairs(expected)
    actual_pairs = _flatten_pairs(actual)
    true_positives = len(expected_pairs & actual_pairs)
    false_positives = len(actual_pairs - expected_pairs)
    false_negatives = len(expected_pairs - actual_pairs)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 1.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact": expected_pairs == actual_pairs,
    }


def run_benchmark(extractor_model: str, artifact_root: Path) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"deterministic": [], "llm": []}

    for case in CASES:
        deterministic = flatten_facts_memory(
            consolidate_facts_memory(DEFAULT_FACTS_MEMORY, case["transcript"], now="2026-04-10T12:00:00Z")
        )
        deterministic_score = compare_flat_facts(case["expected"], deterministic)
        results["deterministic"].append(
            {
                "name": case["name"],
                "expected": case["expected"],
                "actual": deterministic,
                "score": deterministic_score,
            }
        )

        try:
            llm_actual = flatten_facts_memory(
                llm_consolidate_facts_memory(
                    DEFAULT_FACTS_MEMORY,
                    case["transcript"],
                    extractor_model=extractor_model,
                    now="2026-04-10T12:00:00Z",
                )
            )
            llm_score = compare_flat_facts(case["expected"], llm_actual)
            llm_result: dict[str, Any] = {
                "name": case["name"],
                "expected": case["expected"],
                "actual": llm_actual,
                "score": llm_score,
            }
        except Exception as error:
            llm_result = {
                "name": case["name"],
                "expected": case["expected"],
                "actual": None,
                "error": str(error),
            }
        results["llm"].append(llm_result)

    summary = {
        "deterministic": summarize_scores(results["deterministic"]),
        "llm": summarize_scores(results["llm"]),
    }

    payload = {"summary": summary, "cases": results}
    (artifact_root / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact_root / "results.md").write_text(
        format_results_markdown(payload),
        encoding="utf-8",
    )
    return payload


def summarize_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [entry["score"] for entry in results if "score" in entry]
    if not scored:
        return {"cases": 0, "exact": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "cases": len(scored),
        "exact": sum(1 for entry in scored if entry["exact"]),
        "precision": round(sum(entry["precision"] for entry in scored) / len(scored), 4),
        "recall": round(sum(entry["recall"] for entry in scored) / len(scored), 4),
        "f1": round(sum(entry["f1"] for entry in scored) / len(scored), 4),
    }


def format_results_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Facts Extractor Benchmark",
        "",
        "| Extractor | Cases | Exact | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, summary in payload["summary"].items():
        lines.append(
            f"| {name} | {summary['cases']} | {summary['exact']} | {summary['precision']} | {summary['recall']} | {summary['f1']} |"
        )
    lines.extend(["", "## Case Results", ""])
    for extractor_name, entries in payload["cases"].items():
        lines.append(f"### {extractor_name}")
        for entry in entries:
            if "score" in entry:
                lines.append(
                    f"- `{entry['name']}`: exact={entry['score']['exact']}, precision={entry['score']['precision']}, recall={entry['score']['recall']}, f1={entry['score']['f1']}"
                )
            else:
                lines.append(f"- `{entry['name']}`: error={entry['error']}")
        lines.append("")
    return "\n".join(lines)


def _flatten_pairs(data: dict[str, Any]) -> set[tuple[str, str, str]]:
    pairs: set[tuple[str, str, str]] = set()
    for category, items in data.items():
        for key, value in items.items():
            pairs.add((category, key, str(value)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark deterministic vs LLM facts extraction.")
    parser.add_argument(
        "--extractor-model",
        default=DEFAULT_EXTRACTOR_MODEL,
        help="Model used for the LLM facts extractor benchmark.",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/extractor_benchmark/sample_run",
        help="Directory where benchmark artifacts are written.",
    )
    args = parser.parse_args()

    payload = run_benchmark(args.extractor_model, Path(args.artifact_root))
    print(format_results_markdown(payload))


if __name__ == "__main__":
    main()
