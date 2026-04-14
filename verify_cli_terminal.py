from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from agent.chat_service import DEFAULT_CHAT_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real CLI restart scenarios against the local terminal surface."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CHAT_MODEL,
        help="Chat model to use for the live terminal verification run.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Per-CLI-session timeout in seconds.",
    )
    parser.add_argument(
        "--keep-memory-root",
        default=None,
        help="Optional directory to keep the generated memory files instead of using a temp dir.",
    )
    return parser


def run_cli_session(
    *,
    memory_type: str,
    user_id: str,
    memory_root: Path,
    prompts: list[str],
    model: str,
    timeout: int,
) -> str:
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
    completed = subprocess.run(
        command,
        input="\n".join([*prompts, "quit", ""]),
        text=True,
        capture_output=True,
        timeout=timeout,
        env=os.environ.copy(),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{memory_type} CLI session failed with code {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )
    return completed.stdout


def last_assistant_reply(transcript: str) -> str:
    matches = re.findall(r"Assistant:\s*(.*?)(?=\nYou:|\Z)", transcript, flags=re.S)
    return matches[-1].strip() if matches else ""


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def assert_contains(reply: str, *terms: str) -> None:
    missing = [term for term in terms if term.lower() not in reply.lower()]
    if missing:
        raise AssertionError(f"Reply missing expected terms {missing!r}: {reply}")


def assert_forgets(reply: str, forbidden_terms: list[str]) -> None:
    lower = reply.lower()
    denial_signals = [
        "don't have any information",
        "do not have any information",
        "don't know",
        "do not know",
        "could you tell me",
        "you'd need to tell me",
    ]
    if any(signal in lower for signal in denial_signals):
        return
    if all(term.lower() in lower for term in forbidden_terms):
        raise AssertionError(f"Reply still looks like recall after restart: {reply}")


def assert_not_contains(reply: str, *terms: str) -> None:
    lower = reply.lower()
    present = [term for term in terms if term.lower() in lower]
    if present:
        raise AssertionError(f"Reply contains unexpected terms {present!r}: {reply}")


def run_verification(memory_root: Path, model: str, timeout: int) -> dict[str, str]:
    results: dict[str, str] = {}

    same_session_none = run_cli_session(
        memory_type="none",
        user_id="cli_none",
        memory_root=memory_root,
        prompts=[
            "My name is Andrew. My preferred fruit is mango.",
            "What is my name and preferred fruit?",
        ],
        model=model,
        timeout=timeout,
    )
    same_session_reply = last_assistant_reply(same_session_none)
    assert_contains(same_session_reply, "Andrew", "mango")
    results["none_same_session"] = same_session_reply

    restart_none = run_cli_session(
        memory_type="none",
        user_id="cli_none",
        memory_root=memory_root,
        prompts=["What is my name and preferred fruit?"],
        model=model,
        timeout=timeout,
    )
    restart_none_reply = last_assistant_reply(restart_none)
    assert_forgets(restart_none_reply, ["Andrew", "mango"])
    results["none_restart"] = restart_none_reply

    run_cli_session(
        memory_type="facts",
        user_id="cli_facts",
        memory_root=memory_root,
        prompts=["My name is Andrew. My preferred fruit is mango."],
        model=model,
        timeout=timeout,
    )
    facts = load_json(memory_root / "cli_facts" / "facts.json")
    assert facts["profile"]["name"]["value"] == "Andrew"
    assert facts["preferences"]["preferred_fruit"]["value"] == "mango"

    restart_facts = run_cli_session(
        memory_type="facts",
        user_id="cli_facts",
        memory_root=memory_root,
        prompts=["What is my name and preferred fruit?"],
        model=model,
        timeout=timeout,
    )
    restart_facts_reply = last_assistant_reply(restart_facts)
    assert_contains(restart_facts_reply, "Andrew", "mango")
    results["facts_restart"] = restart_facts_reply

    run_cli_session(
        memory_type="summary",
        user_id="cli_summary",
        memory_root=memory_root,
        prompts=["My name is Andrew. My preferred fruit is mango."],
        model=model,
        timeout=timeout,
    )
    summary = load_json(memory_root / "cli_summary" / "summary.json")
    assert_contains(summary["summary"], "Andrew", "mango")

    restart_summary = run_cli_session(
        memory_type="summary",
        user_id="cli_summary",
        memory_root=memory_root,
        prompts=["What is my name and preferred fruit?"],
        model=model,
        timeout=timeout,
    )
    restart_summary_reply = last_assistant_reply(restart_summary)
    assert_contains(restart_summary_reply, "Andrew", "mango")
    results["summary_restart"] = restart_summary_reply

    run_cli_session(
        memory_type="facts",
        user_id="cli_department",
        memory_root=memory_root,
        prompts=[
            "I work in Regulatory Affairs.",
            "Actually, I just transferred to Quality Assurance.",
        ],
        model=model,
        timeout=timeout,
    )
    department_facts = load_json(memory_root / "cli_department" / "facts.json")
    assert department_facts["profile"]["department"]["value"] == "Quality Assurance"

    restart_department = run_cli_session(
        memory_type="facts",
        user_id="cli_department",
        memory_root=memory_root,
        prompts=["What department am I in now?"],
        model=model,
        timeout=timeout,
    )
    restart_department_reply = last_assistant_reply(restart_department)
    assert_contains(restart_department_reply, "Quality Assurance")
    assert_not_contains(restart_department_reply, "doesn't specify a department")
    results["facts_department_restart"] = restart_department_reply

    historical_department = run_cli_session(
        memory_type="facts",
        user_id="cli_department",
        memory_root=memory_root,
        prompts=["What was my job before?"],
        model=model,
        timeout=timeout,
    )
    historical_department_reply = last_assistant_reply(historical_department)
    assert_contains(historical_department_reply, "Regulatory Affairs")
    results["facts_department_history"] = historical_department_reply

    run_cli_session(
        memory_type="facts",
        user_id="cli_profile_updates",
        memory_root=memory_root,
        prompts=[
            (
                "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a medical "
                "device company, focusing on 510(k) submissions for cardiac devices."
            ),
            "Now I focus on 510(k) submissions for brain implants.",
            "I got married and my last name is now Doe.",
        ],
        model=model,
        timeout=timeout,
    )
    updated_profile_facts = load_json(memory_root / "cli_profile_updates" / "facts.json")
    assert updated_profile_facts["profile"]["domain"]["value"] == (
        "510(k) submissions for brain implants"
    )
    assert updated_profile_facts["profile"]["name"]["value"] == "Dr. Sarah Doe"

    updated_profile_events = (memory_root / "cli_profile_updates" / "facts_events.jsonl").read_text()
    assert "cardiac 510(k) submissions" in updated_profile_events
    assert "510(k) submissions for brain implants" in updated_profile_events
    assert "Dr. Sarah Chen" in updated_profile_events
    assert "Dr. Sarah Doe" in updated_profile_events

    restart_profile_updates = run_cli_session(
        memory_type="facts",
        user_id="cli_profile_updates",
        memory_root=memory_root,
        prompts=["What is my name and current focus?"],
        model=model,
        timeout=timeout,
    )
    restart_profile_updates_reply = last_assistant_reply(restart_profile_updates)
    assert_contains(restart_profile_updates_reply, "Sarah Doe", "brain implants")
    results["facts_profile_updates_restart"] = restart_profile_updates_reply

    prior_focus = run_cli_session(
        memory_type="facts",
        user_id="cli_profile_updates",
        memory_root=memory_root,
        prompts=["What did I focus on before my current focus area?"],
        model=model,
        timeout=timeout,
    )
    prior_focus_reply = last_assistant_reply(prior_focus)
    assert_contains(prior_focus_reply, "cardiac")
    updated_profile_facts_after_history = load_json(
        memory_root / "cli_profile_updates" / "facts.json"
    )
    assert updated_profile_facts_after_history["profile"]["domain"]["value"] == (
        "510(k) submissions for brain implants"
    )
    updated_profile_events_after_history = (
        memory_root / "cli_profile_updates" / "facts_events.jsonl"
    ).read_text()
    assert "What did I focus on before my current focus area?" not in (
        updated_profile_events_after_history
    )
    results["facts_profile_updates_history"] = prior_focus_reply

    return results


def main() -> int:
    args = build_parser().parse_args()

    if args.keep_memory_root:
        root = Path(args.keep_memory_root)
        root.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="cli-terminal-verify-")
        root = Path(temp_dir.name)
        cleanup = True

    try:
        results = run_verification(root, args.model, args.timeout)
    except Exception as exc:  # pragma: no cover - live verification path
        print(f"CLI terminal verification failed: {exc}", file=sys.stderr)
        if root.exists():
            print(f"Memory root kept at: {root}", file=sys.stderr)
        return 1

    print("CLI terminal verification passed.")
    print(f"Memory root: {root}")
    print()
    print("Observed replies:")
    for key, value in results.items():
        print(f"- {key}: {value}")

    if cleanup:
        temp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
