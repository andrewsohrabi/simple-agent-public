from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from agent.chat_service import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_FACTS_EXTRACTOR,
)
from agent.memory import (
    facts_events_path,
    facts_path,
    summary_facts_events_path,
    summary_path,
)


STRATEGIES = ("none", "facts", "summary")

SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "identity_recall",
        "sessions": [
            [
                (
                    "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a "
                    "medical device company, focusing on 510(k) submissions for "
                    "cardiac devices."
                ),
                "I got married and my last name is now Doe.",
                "Now I focus on 510(k) submissions for brain implants.",
            ],
            [
                "What do you remember about me and my work now?",
                "What did I focus on before my current focus area?",
            ],
        ],
        "expectations": {
            "none": {
                "checks": [
                    {"scorer": "forgetful"},
                    {"scorer": "forgetful"},
                ],
                "artifact_checks": [{"kind": "no_memory_files"}],
            },
            "facts": {
                "checks": [
                    {
                        "scorer": "terms",
                        "expected_terms": ["Sarah", "Doe", "Regulatory", "brain implant"],
                    },
                    {"scorer": "terms", "expected_terms": ["cardiac", "510(k)"]},
                ],
                "artifact_checks": [
                    {
                        "kind": "memory_contains",
                        "terms": ["Dr. Sarah Doe", "Regulatory Affairs", "brain implants"],
                    },
                    {
                        "kind": "memory_not_contains",
                        "terms": ["cardiac 510(k) submissions"],
                    },
                    {
                        "kind": "events_contain",
                        "terms": [
                            "Dr. Sarah Chen",
                            "Dr. Sarah Doe",
                            "cardiac 510(k) submissions",
                            "510(k) submissions for brain implants",
                        ],
                    },
                ],
            },
            "summary": {
                "checks": [
                    {
                        "scorer": "terms",
                        "expected_terms": ["Sarah", "Doe", "Regulatory", "brain implant"],
                    },
                    {"scorer": "terms", "expected_terms": ["cardiac", "510(k)"]},
                ],
                "artifact_checks": [
                    {
                        "kind": "memory_contains",
                        "terms": ["Dr. Sarah Doe", "Regulatory Affairs", "brain implants"],
                    },
                    {
                        "kind": "memory_not_contains",
                        "terms": ["cardiac 510(k) submissions"],
                    },
                    {
                        "kind": "events_contain",
                        "terms": [
                            "Dr. Sarah Chen",
                            "Dr. Sarah Doe",
                            "cardiac 510(k) submissions",
                            "510(k) submissions for brain implants",
                        ],
                    },
                ],
            },
        },
    },
    {
        "name": "preference_application",
        "sessions": [
            [
                "Going forward, always give me three-line haiku answers.",
                "Update that: start every answer with an ALL-CAPS summary line, then give me concise bullet-point answers. No haikus.",
            ],
            [
                "Explain the key considerations for predicate device selection in a 510(k).",
                "What response style did I ask for before my current one?",
            ],
        ],
        "expectations": {
            "none": {
                "checks": [
                    {"scorer": "forgetful"},
                    {"scorer": "forgetful"},
                ],
                "artifact_checks": [{"kind": "no_memory_files"}],
            },
            "facts": {
                "checks": [
                    {"scorer": "style"},
                    {"scorer": "terms", "expected_terms": ["haiku"]},
                ],
                "artifact_checks": [
                    {
                        "kind": "memory_contains",
                        "terms": ["ALL-CAPS summary line", "concise bullet-point answers"],
                    },
                    {
                        "kind": "memory_not_contains",
                        "terms": ["three-line haiku answers"],
                    },
                    {
                        "kind": "events_contain",
                        "terms": ["three-line haiku answers", "ALL-CAPS summary line"],
                    },
                ],
            },
            "summary": {
                "checks": [
                    {"scorer": "style"},
                    {"scorer": "terms", "expected_terms": ["haiku"]},
                ],
                "artifact_checks": [
                    {
                        "kind": "memory_contains",
                        "terms": ["ALL-CAPS summary line", "concise bullet-point answers"],
                    },
                    {
                        "kind": "memory_not_contains",
                        "terms": ["three-line haiku answers"],
                    },
                    {
                        "kind": "events_contain",
                        "terms": ["three-line haiku answers", "ALL-CAPS summary line"],
                    },
                ],
            },
        },
    },
    {
        "name": "project_context_recall",
        "sessions": [
            [
                (
                    "I'm working on a 510(k) for a new catheter. The main challenge "
                    "is choosing between two predicate devices."
                ),
                (
                    "Now I'm working on a 510(k) for a brain implant. The main "
                    "challenge is building the clinical evidence plan."
                ),
            ],
            [
                "Can you help me think through next steps for my current project?",
                "What project was I working on before my current one?",
            ],
        ],
        "expectations": {
            "none": {
                "checks": [
                    {"scorer": "forgetful"},
                    {"scorer": "forgetful"},
                ],
                "artifact_checks": [{"kind": "no_memory_files"}],
            },
            "facts": {
                "checks": [
                    {
                        "scorer": "terms",
                        "expected_terms": ["brain implant", "clinical evidence"],
                    },
                    {"scorer": "terms", "expected_terms": ["catheter", "predicate"]},
                ],
                "artifact_checks": [
                    {
                        "kind": "memory_contains",
                        "terms": ["brain implant", "clinical evidence plan"],
                    },
                    {
                        "kind": "memory_not_contains",
                        "terms": ["new catheter 510(k)", "predicate devices"],
                    },
                    {
                        "kind": "events_contain",
                        "terms": ["new catheter 510(k)", "brain implant", "predicate devices", "clinical evidence plan"],
                    },
                ],
            },
            "summary": {
                "checks": [
                    {
                        "scorer": "terms",
                        "expected_terms": ["brain implant", "clinical evidence"],
                    },
                    {"scorer": "terms", "expected_terms": ["catheter", "predicate"]},
                ],
                "artifact_checks": [
                    {
                        "kind": "memory_contains",
                        "terms": ["brain implant", "clinical evidence plan"],
                    },
                    {
                        "kind": "memory_not_contains",
                        "terms": ["new catheter 510(k)", "predicate devices"],
                    },
                    {
                        "kind": "events_contain",
                        "terms": ["new catheter 510(k)", "brain implant", "predicate devices", "clinical evidence plan"],
                    },
                ],
            },
        },
    },
    {
        "name": "contradiction_update",
        "sessions": [
            [
                "I work in Regulatory Affairs.",
                "Actually, I just transferred to Quality Assurance.",
            ],
            ["What department am I in now?", "What was my job before?"],
        ],
        "expectations": {
            "none": {
                "checks": [
                    {"scorer": "forgetful"},
                    {"scorer": "forgetful"},
                ],
                "artifact_checks": [{"kind": "no_memory_files"}],
            },
            "facts": {
                "checks": [
                    {"scorer": "terms", "expected_terms": ["Quality Assurance"]},
                    {"scorer": "terms", "expected_terms": ["Regulatory Affairs"]},
                ],
                "artifact_checks": [
                    {"kind": "memory_contains", "terms": ["Quality Assurance"]},
                    {"kind": "memory_not_contains", "terms": ['"Regulatory Affairs"']},
                    {
                        "kind": "events_contain",
                        "terms": ["Regulatory Affairs", "Quality Assurance"],
                    },
                ],
            },
            "summary": {
                "checks": [
                    {"scorer": "terms", "expected_terms": ["Quality Assurance"]},
                    {"scorer": "terms", "expected_terms": ["Regulatory Affairs"]},
                ],
                "artifact_checks": [
                    {"kind": "memory_contains", "terms": ["Quality Assurance"]},
                    {"kind": "memory_not_contains", "terms": ["Regulatory Affairs"]},
                    {
                        "kind": "events_contain",
                        "terms": ["Regulatory Affairs", "Quality Assurance"],
                    },
                ],
            },
        },
    },
    {
        "name": "personal_preference_recall",
        "sessions": [
            [
                "My name is Andrew. My preferred fruit is mango.",
                "Actually, my preferred fruit is pear now.",
            ],
            [
                "What is my name and preferred fruit now?",
                "What fruit did I prefer before my current one?",
            ],
        ],
        "expectations": {
            "none": {
                "checks": [
                    {"scorer": "forgetful"},
                    {"scorer": "forgetful"},
                ],
                "artifact_checks": [{"kind": "no_memory_files"}],
            },
            "facts": {
                "checks": [
                    {"scorer": "terms", "expected_terms": ["Andrew", "pear"]},
                    {"scorer": "terms", "expected_terms": ["mango"]},
                ],
                "artifact_checks": [
                    {"kind": "memory_contains", "terms": ["Andrew", "pear"]},
                    {"kind": "memory_not_contains", "terms": ["mango"]},
                    {"kind": "events_contain", "terms": ["mango", "pear"]},
                ],
            },
            "summary": {
                "checks": [
                    {"scorer": "terms", "expected_terms": ["Andrew", "pear"]},
                    {"scorer": "terms", "expected_terms": ["mango"]},
                ],
                "artifact_checks": [
                    {"kind": "memory_contains", "terms": ["Andrew", "pear"]},
                    {"kind": "memory_not_contains", "terms": ["mango"]},
                    {"kind": "events_contain", "terms": ["mango", "pear"]},
                ],
            },
        },
    },
]
DEFAULT_HARNESS_JOBS = len(STRATEGIES) * len(SCENARIOS)


def build_cli_command(
    *,
    memory_type: str,
    user_id: str,
    memory_root: Path,
    model: str,
    facts_extractor: str,
    extractor_model: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "agent.cli",
        "--model",
        model,
        "--memory-type",
        memory_type,
        "--user-id",
        user_id,
        "--memory-root",
        str(memory_root),
    ]
    if memory_type == "facts":
        command.extend(
            [
                "--facts-extractor",
                facts_extractor,
                "--extractor-model",
                extractor_model,
            ]
        )
    return command


def artifact_paths(root: Path, strategy: str, scenario_name: str) -> dict[str, Path]:
    scenario_dir = root / strategy / scenario_name
    return {
        "dir": scenario_dir,
        "session_1": scenario_dir / "session_1.md",
        "memory_after_session_1": scenario_dir / "memory_after_session_1.json",
        "session_2": scenario_dir / "session_2.md",
        "memory_after_session_2": scenario_dir / "memory_after_session_2.json",
    }


def score_output(output: str, expected_terms: list[str]) -> str:
    output_lower = output.lower()
    matches = sum(1 for term in expected_terms if term.lower() in output_lower)
    if matches == len(expected_terms):
        return "pass"
    if matches:
        return "partial"
    return "fail"


def score_style_output(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "fail"

    first_line = lines[0]
    has_upper_summary = any(character.isalpha() for character in first_line) and first_line == first_line.upper()
    bullet_count = sum(
        1 for line in lines[1:] if line.startswith(("- ", "* ")) or re_numbered(line)
    )
    word_count = len(output.split())
    if has_upper_summary and bullet_count >= 3 and word_count <= 260:
        return "pass"
    if has_upper_summary and bullet_count >= 2:
        return "partial"
    return "fail"


def score_forgetful_output(output: str) -> str:
    lowered = output.lower()
    if any(
        phrase in lowered
        for phrase in (
            "don't know",
            "do not know",
            "don't have any information",
            "do not have any information",
            "don't have information",
            "do not have information",
            "not sure",
            "can't recall",
            "cannot recall",
        )
    ):
        return "pass"
    return "fail"


def extract_assistant_replies(output: str) -> list[str]:
    return [
        reply.strip()
        for reply in re.findall(r"Assistant:\s*(.*?)(?=\nYou:|\Z)", output, flags=re.S)
    ]


def score_check_output(output: str, check: dict[str, Any]) -> str:
    scorer = check.get("scorer")
    if scorer == "style":
        return score_style_output(output)
    if scorer == "forgetful":
        return score_forgetful_output(output)
    return score_output(output, list(check.get("expected_terms", [])))


def aggregate_outcomes(outcomes: list[str]) -> str:
    if not outcomes:
        return "pass"
    if any(outcome == "fail" for outcome in outcomes):
        return "fail"
    if all(outcome == "pass" for outcome in outcomes):
        return "pass"
    return "partial"


def audit_artifacts(
    *,
    strategy: str,
    scenario: dict[str, Any],
    memory_root: Path,
    user_id: str,
) -> dict[str, Any]:
    expectation = scenario["expectations"][strategy]
    checks: list[dict[str, Any]] = []

    if strategy == "none":
        relevant_paths = [
            facts_path(memory_root, user_id),
            facts_events_path(memory_root, user_id),
            summary_path(memory_root, user_id),
            summary_facts_events_path(memory_root, user_id),
        ]
        no_files = not any(path.exists() for path in relevant_paths)
        checks.append(
            {
                "label": "no persisted memory artifacts",
                "outcome": "pass" if no_files else "fail",
            }
        )
        return {"outcome": aggregate_outcomes([check["outcome"] for check in checks]), "checks": checks}

    memory_text = _memory_surface_text(strategy, memory_root, user_id)
    events_text = _events_surface_text(strategy, memory_root, user_id)

    for check in expectation.get("artifact_checks", []):
        kind = check["kind"]
        terms = list(check.get("terms", []))
        if kind == "memory_contains":
            outcome = "pass" if all(term in memory_text for term in terms) else "fail"
            label = f"memory contains {', '.join(terms)}"
        elif kind == "memory_not_contains":
            outcome = "pass" if all(term not in memory_text for term in terms) else "fail"
            label = f"memory omits {', '.join(terms)}"
        elif kind == "events_contain":
            outcome = "pass" if all(term in events_text for term in terms) else "fail"
            label = f"events contain {', '.join(terms)}"
        elif kind == "no_memory_files":
            outcome = "pass" if not memory_text and not events_text else "fail"
            label = "no persisted memory artifacts"
        else:
            outcome = "fail"
            label = f"unknown audit kind: {kind}"
        checks.append({"label": label, "outcome": outcome})

    return {
        "outcome": aggregate_outcomes([check["outcome"] for check in checks]),
        "checks": checks,
    }


def run_harness(
    *,
    model: str,
    memory_root: Path,
    artifact_root: Path,
    base_user_id: str,
    facts_extractor: str,
    extractor_model: str,
    stream_output: bool,
    jobs: int,
) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "config": {
            "model": model,
            "facts_extractor": facts_extractor,
            "extractor_model": extractor_model,
            "jobs": jobs,
            "strategies": list(STRATEGIES),
            "scenarios": [scenario["name"] for scenario in SCENARIOS],
        },
        "scenarios": {
            scenario["name"]: {strategy: {} for strategy in STRATEGIES}
            for scenario in SCENARIOS
        },
    }
    print_lock = threading.Lock()

    if stream_output:
        print(_format_run_header(results["config"]), flush=True)

    tasks = []
    for scenario_index, scenario in enumerate(SCENARIOS, start=1):
        for strategy in STRATEGIES:
            tasks.append(
                {
                    "scenario": scenario,
                    "scenario_index": scenario_index,
                    "strategy": strategy,
                    "user_id": f"{base_user_id}_{scenario['name']}_{strategy}",
                }
            )

    max_workers = max(1, min(jobs, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_scenario_strategy,
                task=task,
                total_scenarios=len(SCENARIOS),
                model=model,
                memory_root=memory_root,
                artifact_root=artifact_root,
                facts_extractor=facts_extractor,
                extractor_model=extractor_model,
                stream_output=stream_output,
                print_lock=print_lock,
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            result = future.result()
            results["scenarios"][result["scenario_name"]][result["strategy"]] = {
                "outcome": result["outcome"],
                "user_id": result["user_id"],
                "session_2_checks": result["session_2_checks"],
                "artifact_audit": result["artifact_audit"],
                "artifacts": result["artifacts"],
            }

    _write_results(artifact_root, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run memory strategy comparison harness.")
    parser.add_argument(
        "--model",
        default=os.environ.get("HARNESS_MODEL", DEFAULT_CHAT_MODEL),
    )
    parser.add_argument("--memory-root", default="memory_store")
    parser.add_argument(
        "--artifact-root", default="artifacts/harness_results/sample_run"
    )
    parser.add_argument("--user-id", default="demo_user")
    parser.add_argument(
        "--facts-extractor",
        default=DEFAULT_FACTS_EXTRACTOR,
        help="Facts extraction mode used for facts strategy sessions.",
    )
    parser.add_argument(
        "--extractor-model",
        default=DEFAULT_EXTRACTOR_MODEL,
        help="Model used for optional facts LLM consolidation in facts mode.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("HARNESS_JOBS", DEFAULT_HARNESS_JOBS)),
        help="Maximum number of scenario/strategy jobs to run in parallel.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-scenario streaming progress output.",
    )
    args = parser.parse_args()

    results = run_harness(
        model=args.model,
        memory_root=Path(args.memory_root),
        artifact_root=Path(args.artifact_root),
        base_user_id=args.user_id,
        facts_extractor=args.facts_extractor,
        extractor_model=args.extractor_model,
        stream_output=not args.quiet,
        jobs=args.jobs,
    )
    print(_format_terminal_summary(results))


def _run_cli_session(
    command: list[str],
    messages: list[str],
    *,
    stream_output: bool,
    line_prefix: str = "",
    print_lock: threading.Lock | None = None,
) -> str:
    process_input = "\n".join([*messages, "exit"]) + "\n"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if not stream_output:
        result = subprocess.run(
            command,
            input=process_input,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if result.returncode != 0:
            return result.stdout + "\n[stderr]\n" + result.stderr
        return result.stdout

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert process.stdin is not None
    assert process.stdout is not None

    process.stdin.write(process_input)
    process.stdin.close()

    output_lines: list[str] = []
    for line in process.stdout:
        output_lines.append(line)
        if print_lock is None:
            print(f"{line_prefix}{line}", end="", flush=True)
        else:
            with print_lock:
                print(f"{line_prefix}{line}", end="", flush=True)

    return_code = process.wait()
    output = "".join(output_lines)
    if return_code != 0:
        return output + f"\n[stderr]\nprocess exited with code {return_code}\n"
    return output


def _run_scenario_strategy(
    *,
    task: dict[str, Any],
    total_scenarios: int,
    model: str,
    memory_root: Path,
    artifact_root: Path,
    facts_extractor: str,
    extractor_model: str,
    stream_output: bool,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    scenario = task["scenario"]
    scenario_name = scenario["name"]
    strategy = task["strategy"]
    user_id = task["user_id"]
    scenario_index = task["scenario_index"]

    _clear_user_memory(memory_root, user_id)
    paths = artifact_paths(artifact_root, strategy, scenario_name)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    if stream_output:
        with print_lock:
            print(
                f"\n[{scenario_index}/{total_scenarios}] {scenario_name} :: {strategy}",
                flush=True,
            )

    session_outputs = []
    for index, session_messages in enumerate(scenario["sessions"], start=1):
        if stream_output:
            prompts = " | ".join(session_messages)
            with print_lock:
                print(f"  session {index}: {prompts}", flush=True)
        output = _run_cli_session(
            build_cli_command(
                memory_type=strategy,
                user_id=user_id,
                memory_root=memory_root,
                model=model,
                facts_extractor=facts_extractor,
                extractor_model=extractor_model,
            ),
            session_messages,
            stream_output=stream_output,
            line_prefix=f"    [{scenario_name}/{strategy}/s{index}] ",
            print_lock=print_lock,
        )
        session_outputs.append(output)
        paths[f"session_{index}"].write_text(
            _format_session_artifact(index, session_messages, output),
            encoding="utf-8",
        )

        if index == 1:
            _write_memory_snapshot(
                paths["memory_after_session_1"],
                memory_root,
                user_id,
                strategy,
            )
        if index == 2:
            _write_memory_snapshot(
                paths["memory_after_session_2"],
                memory_root,
                user_id,
                strategy,
            )

    session_2_replies = extract_assistant_replies(session_outputs[-1])
    prompt_checks = scenario["expectations"][strategy]["checks"]
    check_outcomes = []
    for index, check in enumerate(prompt_checks):
        reply = session_2_replies[index] if index < len(session_2_replies) else ""
        check_outcomes.append(
            {
                "prompt": scenario["sessions"][1][index],
                "reply": reply,
                "outcome": score_check_output(reply, check),
            }
        )

    artifact_audit = audit_artifacts(
        strategy=strategy,
        scenario=scenario,
        memory_root=memory_root,
        user_id=user_id,
    )
    outcome = aggregate_outcomes(
        [check["outcome"] for check in check_outcomes] + [artifact_audit["outcome"]]
    )

    if stream_output:
        with print_lock:
            print(f"  outcome: {outcome}", flush=True)

    return {
        "scenario_name": scenario_name,
        "strategy": strategy,
        "outcome": outcome,
        "user_id": user_id,
        "session_2_checks": check_outcomes,
        "artifact_audit": artifact_audit,
        "artifacts": {
            "session_1": str(paths["session_1"]),
            "memory_after_session_1": str(paths["memory_after_session_1"]),
            "session_2": str(paths["session_2"]),
            "memory_after_session_2": str(paths["memory_after_session_2"]),
        },
    }


def _clear_user_memory(memory_root: Path, user_id: str) -> None:
    shutil.rmtree(memory_root / user_id, ignore_errors=True)
    shutil.rmtree(memory_root / f"{user_id}__summary_internal", ignore_errors=True)


def _write_memory_snapshot(
    path: Path, memory_root: Path, user_id: str, strategy: str
) -> None:
    if strategy == "facts":
        snapshot = {
            "memory": _load_json_if_exists(facts_path(memory_root, user_id)),
            "facts_events": _load_jsonl_if_exists(facts_events_path(memory_root, user_id)),
        }
    elif strategy == "summary":
        snapshot = {
            "memory": _load_json_if_exists(summary_path(memory_root, user_id)),
            "temporal_events": _load_jsonl_if_exists(
                summary_facts_events_path(memory_root, user_id)
            ),
        }
    else:
        snapshot = {"memory": None}

    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _memory_surface_text(strategy: str, memory_root: Path, user_id: str) -> str:
    if strategy == "facts":
        payload = _load_json_if_exists(facts_path(memory_root, user_id))
        if not isinstance(payload, dict):
            return ""
        values = []
        for category in payload.values():
            if not isinstance(category, dict):
                continue
            for fact in category.values():
                if isinstance(fact, dict) and fact.get("value"):
                    values.append(str(fact["value"]))
        return "\n".join(values)

    if strategy == "summary":
        payload = _load_json_if_exists(summary_path(memory_root, user_id))
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("summary") or "")

    return ""


def _events_surface_text(strategy: str, memory_root: Path, user_id: str) -> str:
    if strategy == "facts":
        events = _load_jsonl_if_exists(facts_events_path(memory_root, user_id))
    elif strategy == "summary":
        events = _load_jsonl_if_exists(summary_facts_events_path(memory_root, user_id))
    else:
        events = []

    lines: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        parts = [
            str(event.get("old_value") or ""),
            str(event.get("new_value") or ""),
            str(event.get("source") or ""),
        ]
        lines.append(" ".join(part for part in parts if part))
    return "\n".join(lines)


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_if_exists(path: Path) -> list[Any]:
    if not path.exists():
        return []
    items: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _format_session_artifact(
    session_index: int, messages: list[str], output: str
) -> str:
    prompts = "\n".join(f"- {message}" for message in messages)
    return f"# Session {session_index}\n\n## Prompts\n\n{prompts}\n\n## CLI Output\n\n```text\n{output}\n```\n"


def _write_results(artifact_root: Path, results: dict[str, Any]) -> None:
    (artifact_root / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact_root / "results.md").write_text(
        _format_results_markdown(results),
        encoding="utf-8",
    )


def _format_results_markdown(results: dict[str, Any]) -> str:
    config = results["config"]
    lines = [
        "# Harness Results",
        "",
        f"- Model: `{config['model']}`",
        f"- Facts extractor: `{config['facts_extractor']}`",
        f"- Facts extractor model: `{config['extractor_model']}`",
        "",
        "| Scenario | none | facts | summary |",
        "| --- | --- | --- | --- |",
    ]
    for scenario_name, scenario_results in results["scenarios"].items():
        lines.append(
            f"| {scenario_name} | "
            f"{scenario_results['none']['outcome']} | "
            f"{scenario_results['facts']['outcome']} | "
            f"{scenario_results['summary']['outcome']} |"
        )
    lines.append("")
    lines.append("## Artifact audits")
    lines.append("")
    for scenario_name, scenario_results in results["scenarios"].items():
        lines.append(f"### {scenario_name}")
        lines.append("")
        for strategy in STRATEGIES:
            audit = scenario_results[strategy].get("artifact_audit", {})
            lines.append(
                f"- `{strategy}`: {audit.get('outcome', 'unknown')}"
            )
            for check in audit.get("checks", []):
                lines.append(f"  - {check['label']}: {check['outcome']}")
        lines.append("")
    return "\n".join(lines)


def _format_terminal_summary(results: dict[str, Any]) -> str:
    config = results["config"]
    lines = [
        "=== Comparison Summary ===",
        f"model={config['model']}",
        f"facts_extractor={config['facts_extractor']}",
        f"extractor_model={config['extractor_model']}",
        f"jobs={config['jobs']}",
    ]
    for scenario_name, scenario_results in results["scenarios"].items():
        outcomes = ", ".join(
            (
                f"{strategy}={scenario_results[strategy]['outcome']}"
                f" (audit={scenario_results[strategy].get('artifact_audit', {}).get('outcome', 'unknown')})"
            )
            for strategy in STRATEGIES
        )
        lines.append(f"- {scenario_name}: {outcomes}")
    return "\n".join(lines)


def _format_run_header(config: dict[str, Any]) -> str:
    scenarios = ", ".join(config["scenarios"])
    strategies = ", ".join(config["strategies"])
    return (
        "=== Running Comparison Harness ===\n"
        f"model={config['model']}\n"
        f"facts_extractor={config['facts_extractor']}\n"
        f"extractor_model={config['extractor_model']}\n"
        f"jobs={config['jobs']}\n"
        f"strategies={strategies}\n"
        f"scenarios={scenarios}"
    )


def re_numbered(line: str) -> bool:
    prefix = line.split(" ", 1)[0]
    return len(prefix) > 1 and prefix[:-1].isdigit() and prefix[-1] in ".)"


if __name__ == "__main__":
    main()
