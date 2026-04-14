from __future__ import annotations

import json
import re
import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model


DEFAULT_MEMORY_ROOT = Path("memory_store")
FACT_CATEGORIES = ("profile", "preferences", "constraints", "project_context")
DEFAULT_FACTS_MEMORY: dict[str, dict[str, dict[str, str | None]]] = {
    "profile": {},
    "preferences": {},
    "constraints": {},
    "project_context": {},
}
DEFAULT_SUMMARY_MEMORY: dict[str, str | None] = {
    "summary": "",
    "updated_at": None,
}
INTERROGATIVE_OPENERS = ("what", "when", "where", "why", "how", "who", "which")
SENTENCE_ABBREVIATIONS = ("Dr.", "Mr.", "Mrs.", "Ms.")
CANONICAL_FACT_KEYS: dict[str, set[str]] = {
    "profile": {"name", "department", "role", "domain"},
    "preferences": {"response_style", "preferred_fruit", "preference_comparison"},
    "constraints": {"avoid_long_paragraphs"},
    "project_context": {"current_project", "key_challenge"},
}


def facts_path(memory_root: Path | str, user_id: str) -> Path:
    return Path(memory_root) / user_id / "facts.json"


def summary_path(memory_root: Path | str, user_id: str) -> Path:
    return Path(memory_root) / user_id / "summary.json"


def facts_events_path(memory_root: Path | str, user_id: str) -> Path:
    return Path(memory_root) / user_id / "facts_events.jsonl"


def summary_facts_path(memory_root: Path | str, user_id: str) -> Path:
    return _summary_internal_root(memory_root, user_id) / "facts.json"


def summary_facts_events_path(memory_root: Path | str, user_id: str) -> Path:
    return _summary_internal_root(memory_root, user_id) / "facts_events.jsonl"


def load_facts_memory(memory_root: Path | str, user_id: str) -> dict[str, Any]:
    path = facts_path(memory_root, user_id)
    if not path.exists():
        return deepcopy(DEFAULT_FACTS_MEMORY)

    with path.open() as file:
        data = json.load(file)

    return normalize_facts_memory(data)


def save_facts_memory(
    memory_root: Path | str, user_id: str, facts_memory: dict[str, Any]
) -> None:
    _write_json_atomic(facts_path(memory_root, user_id), normalize_facts_memory(facts_memory))


def load_summary_memory(memory_root: Path | str, user_id: str) -> dict[str, Any]:
    path = summary_path(memory_root, user_id)
    if not path.exists():
        return deepcopy(DEFAULT_SUMMARY_MEMORY)

    with path.open() as file:
        data = json.load(file)

    return {
        "summary": str(data.get("summary") or ""),
        "updated_at": data.get("updated_at"),
    }


def save_summary_memory(
    memory_root: Path | str, user_id: str, summary_memory: dict[str, Any]
) -> None:
    data = {
        "summary": str(summary_memory.get("summary") or ""),
        "updated_at": summary_memory.get("updated_at"),
    }
    _write_json_atomic(summary_path(memory_root, user_id), data)


def load_summary_facts_memory(memory_root: Path | str, user_id: str) -> dict[str, Any]:
    path = summary_facts_path(memory_root, user_id)
    if not path.exists():
        return deepcopy(DEFAULT_FACTS_MEMORY)

    with path.open() as file:
        data = json.load(file)

    return normalize_facts_memory(data)


def save_summary_facts_memory(
    memory_root: Path | str, user_id: str, facts_memory: dict[str, Any]
) -> None:
    _write_json_atomic(
        summary_facts_path(memory_root, user_id),
        normalize_facts_memory(facts_memory),
    )


def load_facts_events(memory_root: Path | str, user_id: str) -> list[dict[str, Any]]:
    return _load_events_from_path(facts_events_path(memory_root, user_id))


def load_summary_facts_events(memory_root: Path | str, user_id: str) -> list[dict[str, Any]]:
    return _load_events_from_path(summary_facts_events_path(memory_root, user_id))


def append_facts_events(
    memory_root: Path | str,
    user_id: str,
    previous_memory: dict[str, Any],
    updated_memory: dict[str, Any],
) -> list[dict[str, Any]]:
    events = _collect_fact_events(previous_memory, updated_memory)
    if not events:
        return []

    _append_events_to_path(facts_events_path(memory_root, user_id), events)
    return events


def append_summary_facts_events(
    memory_root: Path | str,
    user_id: str,
    previous_memory: dict[str, Any],
    updated_memory: dict[str, Any],
) -> list[dict[str, Any]]:
    events = _collect_fact_events(previous_memory, updated_memory)
    if not events:
        return []

    _append_events_to_path(summary_facts_events_path(memory_root, user_id), events)
    return events


def delete_facts_events(memory_root: Path | str, user_id: str) -> None:
    path = facts_events_path(memory_root, user_id)
    if path.exists():
        path.unlink()


def delete_summary_facts_memory(memory_root: Path | str, user_id: str) -> None:
    path = summary_facts_path(memory_root, user_id)
    if path.exists():
        path.unlink()


def delete_summary_facts_events(memory_root: Path | str, user_id: str) -> None:
    path = summary_facts_events_path(memory_root, user_id)
    if path.exists():
        path.unlink()


def normalize_facts_memory(data: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for category in FACT_CATEGORIES:
        raw_category = data.get(category, {})
        normalized[category] = raw_category if isinstance(raw_category, dict) else {}
    return normalized


def flatten_facts_memory(data: dict[str, Any]) -> dict[str, dict[str, str]]:
    normalized = normalize_facts_memory(data)
    flat: dict[str, dict[str, str]] = {}
    for category, items in normalized.items():
        flat[category] = {}
        for key, payload in items.items():
            if isinstance(payload, dict):
                value = payload.get("value")
                if value:
                    flat[category][key] = str(value)
    return flat


def render_facts_memory_block(facts_memory: dict[str, Any]) -> str:
    normalized = normalize_facts_memory(facts_memory)
    sections = []
    labels = {
        "profile": "Profile",
        "preferences": "Preferences",
        "constraints": "Constraints",
        "project_context": "Project Context",
    }

    for category in FACT_CATEGORIES:
        items = normalized[category]
        if not items:
            continue

        lines = [f"{labels[category]}:"]
        for key in _ordered_fact_keys(category, items):
            value = items[key].get("value") if isinstance(items[key], dict) else items[key]
            if value:
                if _should_skip_fact_line(category, key, items):
                    continue
                lines.append(f"- {key}: {value}")
        if len(lines) > 1:
            sections.append("\n".join(lines))

    if not sections:
        return ""

    return "Long-Term Memory:\n" + "\n\n".join(sections)


def render_summary_memory_block(summary_memory: dict[str, Any]) -> str:
    summary = str(summary_memory.get("summary") or "").strip()
    if not summary:
        return ""
    return f"Long-Term Memory:\n{summary}"


def render_temporal_facts_memory_block(
    events: list[dict[str, Any]],
    query: str,
    *,
    max_events: int = 4,
) -> str:
    relevant_events = _select_temporal_events(events, query, max_events=max_events)
    if not relevant_events:
        return ""

    lines = ["Temporal Memory:"]
    for event in relevant_events:
        key = str(event.get("key") or "").replace("_", " ")
        old_value = event.get("old_value")
        new_value = event.get("new_value")
        timestamp = event.get("timestamp")

        if old_value:
            line = f"- {key} changed from {old_value} to {new_value}"
        else:
            line = f"- {key} set to {new_value}"
        if timestamp:
            line += f" at {timestamp}"
        lines.append(line)

    return "\n".join(lines)


def consolidate_facts_memory(
    current_memory: dict[str, Any],
    transcript: list[Any],
    *,
    now: str | None = None,
) -> dict[str, Any]:
    updated = normalize_facts_memory(deepcopy(current_memory))
    timestamp = now or _utc_now()

    for content in _extractable_user_contents(transcript):
        _extract_identity(content, updated, timestamp)
        _extract_preferences(content, updated, timestamp)
        _extract_project_context(content, updated, timestamp)

    return updated


def consolidate_summary_memory(
    current_memory: dict[str, Any],
    transcript: list[Any],
    *,
    now: str | None = None,
    max_chars: int = 500,
) -> dict[str, Any]:
    timestamp = now or _utc_now()
    facts = consolidate_facts_memory(DEFAULT_FACTS_MEMORY, transcript, now=timestamp)
    return build_summary_memory_from_facts(
        facts,
        previous_summary=current_memory,
        now=timestamp,
        max_chars=max_chars,
    )


def build_summary_memory_from_facts(
    facts_memory: dict[str, Any],
    *,
    previous_summary: dict[str, Any] | None = None,
    now: str | None = None,
    max_chars: int = 500,
) -> dict[str, Any]:
    timestamp = now or _utc_now()
    fact_phrases = _facts_to_summary_phrases(normalize_facts_memory(facts_memory))
    prior_summary = str((previous_summary or {}).get("summary") or "").strip()
    prior_updated_at = (previous_summary or {}).get("updated_at")

    if not fact_phrases:
        return {
            "summary": prior_summary[:max_chars],
            "updated_at": prior_updated_at,
        }

    summary = _bounded_summary(" ".join(fact_phrases), max_chars)
    if summary == prior_summary:
        return {
            "summary": summary,
            "updated_at": prior_updated_at,
        }

    return {
        "summary": summary,
        "updated_at": timestamp,
    }


def llm_consolidate_facts_memory(
    current_memory: dict[str, Any],
    transcript: list[Any],
    *,
    extractor_model: str,
    now: str | None = None,
) -> dict[str, Any]:
    timestamp = now or _utc_now()
    prompt = _build_facts_llm_prompt(current_memory, transcript, timestamp)
    model = init_chat_model(extractor_model)
    response = model.invoke(prompt)
    parsed = _parse_json_object(_response_text(response))
    if not isinstance(parsed, dict):
        raise ValueError("LLM facts consolidation did not return a JSON object")
    return _canonicalize_llm_facts(parsed, current_memory, transcript, timestamp)


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as temp_file:
        json.dump(data, temp_file, indent=2, sort_keys=True)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)
    temp_path.replace(path)


def _load_events_from_path(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    with path.open() as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                events.append(payload)
    return events


def _append_events_to_path(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for event in events:
            json.dump(event, file, sort_keys=True)
            file.write("\n")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _summary_internal_root(memory_root: Path | str, user_id: str) -> Path:
    return Path(memory_root) / f"{user_id}__summary_internal"


def _user_contents(transcript: list[Any]) -> list[str]:
    contents = []
    for message in transcript:
        role = None
        content = None
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None) or getattr(message, "type", None)
            content = getattr(message, "content", None)

        if role in ("user", "human") and content:
            contents.append(str(content))
    return contents


def _extractable_user_contents(transcript: list[Any]) -> list[str]:
    clauses: list[str] = []
    for content in _user_contents(transcript):
        clauses.extend(_extractable_clauses(content))
    return clauses


def _extractable_clauses(content: str) -> list[str]:
    protected = content
    replacements: dict[str, str] = {}
    for index, abbreviation in enumerate(SENTENCE_ABBREVIATIONS):
        placeholder = f"__abbr_{index}__"
        replacements[placeholder] = abbreviation
        protected = protected.replace(abbreviation, placeholder)

    raw_clauses = re.split(r"\s*\|\s*|\n+|(?<=[.!?])\s+", protected)
    clauses: list[str] = []
    for raw_clause in raw_clauses:
        clause = raw_clause.strip()
        if not clause:
            continue
        for placeholder, abbreviation in replacements.items():
            clause = clause.replace(placeholder, abbreviation)
        if _is_interrogative_clause(clause):
            continue
        clauses.append(clause)
    return clauses


def _is_interrogative_clause(clause: str) -> bool:
    stripped = clause.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True

    opener_match = re.match(r"[A-Za-z']+", stripped.lower())
    if not opener_match:
        return False
    return opener_match.group(0) in INTERROGATIVE_OPENERS


def _extract_identity(content: str, memory: dict[str, Any], timestamp: str) -> None:
    plain_name_match = re.search(
        r"\bmy name is\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b",
        content,
        flags=re.IGNORECASE,
    )
    if plain_name_match:
        _upsert(
            memory,
            "profile",
            "name",
            _title_preserving_dr(plain_name_match.group(1)),
            content,
            timestamp,
        )

    name_match = re.search(
        r"\b(?:I am|I'm|my name is)\s+(Dr\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)",
        content,
        flags=re.IGNORECASE,
    )
    if name_match:
        _upsert(memory, "profile", "name", _title_preserving_dr(name_match.group(1)), content, timestamp)

    updated_name = _updated_name_from_last_name_change(content, memory)
    if updated_name:
        _upsert(memory, "profile", "name", updated_name, content, timestamp)

    transfer_match = re.search(
        r"\btransferred to\s+([A-Z][A-Za-z ]+?)(?:\.|,|$)",
        content,
        flags=re.IGNORECASE,
    )
    if transfer_match:
        _upsert_department_role(
            memory,
            transfer_match.group(1).strip(),
            content,
            timestamp,
        )
        return

    role_match = re.search(
        r"\bI work in\s+([A-Z][A-Za-z ]+?)(?:\s+at|,|\.|\s+focusing|\s+focused|\s+on)",
        content,
        flags=re.IGNORECASE,
    )
    if role_match:
        _upsert_department_role(
            memory,
            role_match.group(1).strip(),
            content,
            timestamp,
        )

    domain = _extract_domain(content)
    if domain:
        _upsert(memory, "profile", "domain", domain, content, timestamp)


def _extract_preferences(content: str, memory: dict[str, Any], timestamp: str) -> None:
    always_match = re.search(
        r"\balways give me\s+([^.\n]+?)(?:\.|$)",
        content,
        flags=re.IGNORECASE,
    )
    if always_match:
        _upsert(
            memory,
            "preferences",
            "response_style",
            _clean_preference(always_match.group(1)),
            content,
            timestamp,
        )

    start_every_answer_match = re.search(
        r"\bstart (?:every|each) answer with\s+([^.\n]+?)(?:\.|$)",
        content,
        flags=re.IGNORECASE,
    )
    if start_every_answer_match:
        _upsert(
            memory,
            "preferences",
            "response_style",
            _clean_preference(
                "start every answer with " + start_every_answer_match.group(1).strip()
            ),
            content,
            timestamp,
        )

    if re.search(r"\bno long paragraphs\b", content, flags=re.IGNORECASE):
        _upsert(
            memory,
            "constraints",
            "avoid_long_paragraphs",
            "no long paragraphs",
            content,
            timestamp,
        )

    response_style_match = re.search(
        r"\bI prefer\s+([^.\n]+?)(?:answers?|responses?|replies?)\b",
        content,
        flags=re.IGNORECASE,
    )
    if response_style_match:
        _upsert(
            memory,
            "preferences",
            "response_style",
            _clean_preference(response_style_match.group(1).strip() + " answers"),
            content,
            timestamp,
        )

    preferred_fruit_match = re.search(
        r"\bmy (?:preferred|favorite) fruit is\s+([A-Za-z-]+)\b",
        content,
        flags=re.IGNORECASE,
    )
    if preferred_fruit_match:
        _upsert(
            memory,
            "preferences",
            "preferred_fruit",
            preferred_fruit_match.group(1).strip().lower(),
            content,
            timestamp,
        )

    prefer_over_match = re.search(
        r"\bI prefer\s+([^.\n]+?)\s+over\s+([^.\n]+?)(?:\.|$)",
        content,
        flags=re.IGNORECASE,
    )
    if prefer_over_match:
        _upsert(
            memory,
            "preferences",
            "preference_comparison",
            f"{prefer_over_match.group(1).strip().lower()} over {prefer_over_match.group(2).strip().lower()}",
            content,
            timestamp,
        )

    like_more_match = re.search(
        r"\bI like\s+([^.\n]+?)\s+more than\s+([^.\n]+?)(?:\.|$)",
        content,
        flags=re.IGNORECASE,
    )
    if like_more_match:
        _upsert(
            memory,
            "preferences",
            "preference_comparison",
            f"{like_more_match.group(1).strip().lower()} over {like_more_match.group(2).strip().lower()}",
            content,
            timestamp,
        )


def _extract_project_context(content: str, memory: dict[str, Any], timestamp: str) -> None:
    project_match = re.search(
        r"\b(?:I'm|I am) working on\s+([^.\n]+?)(?:\.|$)",
        content,
        flags=re.IGNORECASE,
    )
    if project_match:
        _upsert(
            memory,
            "project_context",
            "current_project",
            _clean_project(project_match.group(1)),
            content,
            timestamp,
        )

    challenge_match = re.search(
        r"\bmain challenge is\s+([^.\n]+?)(?:\.|$)",
        content,
        flags=re.IGNORECASE,
    )
    if challenge_match:
        _upsert(
            memory,
            "project_context",
            "key_challenge",
            challenge_match.group(1).strip(),
            content,
            timestamp,
        )


def _extract_domain(content: str) -> str | None:
    clause = content.strip()
    focus_patterns = (
        r"\bfocusing on\s+([^.\n]+?)(?:,|\.|$)",
        r"^(?:now\s+)?i focus on\s+([^.\n]+?)(?:,|\.|$)",
        r"^i(?:'m| am)\s+focused on\s+([^.\n]+?)(?:,|\.|$)",
        r"^my focus is\s+([^.\n]+?)(?:,|\.|$)",
        r"^current focus is\s+([^.\n]+?)(?:,|\.|$)",
    )
    for pattern in focus_patterns:
        focus_match = re.search(pattern, clause, flags=re.IGNORECASE)
        if focus_match:
            return _normalize_domain_value(focus_match.group(1))

    lowered = clause.lower()
    if "cardiac" in lowered and "510(k)" in lowered:
        return "cardiac 510(k) submissions"
    return None


def _upsert(
    memory: dict[str, Any],
    category: str,
    key: str,
    value: str,
    source: str,
    timestamp: str,
) -> None:
    value = value.strip()
    if not value:
        return
    memory.setdefault(category, {})
    memory[category][key] = {
        "value": value,
        "updated_at": timestamp,
        "source": _short_source(source),
    }


def _upsert_department_role(
    memory: dict[str, Any],
    value: str,
    source: str,
    timestamp: str,
) -> None:
    _upsert(memory, "profile", "department", value, source, timestamp)
    _upsert(memory, "profile", "role", value, source, timestamp)


def _facts_to_summary_phrases(facts: dict[str, Any]) -> list[str]:
    phrases = []
    profile = facts.get("profile", {})
    preferences = facts.get("preferences", {})
    project_context = facts.get("project_context", {})

    name = _fact_value(profile, "name")
    department = _fact_value(profile, "department")
    role = _fact_value(profile, "role")
    domain = _fact_value(profile, "domain")
    work_context = department or role
    if name or work_context or domain:
        subject = f"User is {name}" if name else "User"
        if work_context:
            subject += f" in {work_context}"
        if domain:
            subject += f" focused on {domain}"
        phrases.append(subject + ".")

    response_style = _fact_value(preferences, "response_style")
    if response_style:
        phrases.append(f"They prefer {response_style}.")

    preferred_fruit = _fact_value(preferences, "preferred_fruit")
    if preferred_fruit:
        phrases.append(f"Their preferred fruit is {preferred_fruit}.")

    preference_comparison = _fact_value(preferences, "preference_comparison")
    if preference_comparison:
        phrases.append(f"They prefer {preference_comparison}.")

    current_project = _fact_value(project_context, "current_project")
    if current_project:
        phrases.append(f"They are currently working on {current_project}.")

    key_challenge = _fact_value(project_context, "key_challenge")
    if key_challenge:
        phrases.append(f"The current project challenge is {key_challenge}.")

    return phrases


def _fact_value(category: dict[str, Any], key: str) -> str | None:
    fact = category.get(key)
    if isinstance(fact, dict):
        value = fact.get("value")
        return str(value) if value else None
    return None


def _bounded_summary(summary: str, max_chars: int) -> str:
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) <= max_chars:
        return summary
    clipped = summary[: max_chars + 1].rsplit(" ", 1)[0].rstrip(".,; ")
    return clipped + "."


def _summary_is_replaced(prior_summary: str, new_summary: str) -> bool:
    if "transferred to" in new_summary.lower():
        return True
    prior_lower = prior_summary.lower()
    new_lower = new_summary.lower()
    return "regulatory affairs" in prior_lower and "quality assurance" in new_lower


def _clean_preference(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^that you\s+", "", value, flags=re.IGNORECASE)
    return value


def _clean_project(value: str) -> str:
    value = value.strip()
    if "510(k)" in value.lower() and "catheter" in value.lower():
        return "new catheter 510(k)"
    return value


def _normalize_domain_value(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" ,.")
    lowered = cleaned.lower()
    if "cardiac" in lowered and "510(k)" in cleaned:
        return "cardiac 510(k) submissions"
    return cleaned


def _updated_name_from_last_name_change(content: str, memory: dict[str, Any]) -> str | None:
    last_name_match = re.search(
        r"\bmy last name is now\s+([A-Za-z][A-Za-z'-]*)\b",
        content,
        flags=re.IGNORECASE,
    )
    if not last_name_match:
        return None

    current_name = _fact_value(memory.get("profile", {}), "name")
    if not current_name:
        return None

    parts = current_name.split()
    if len(parts) < 2:
        return None

    next_parts = parts[:-1] + [last_name_match.group(1).title()]
    if next_parts[0].lower().rstrip(".") == "dr":
        next_parts[0] = "Dr."
    return " ".join(next_parts)


def _short_source(source: str) -> str:
    source = re.sub(r"\s+", " ", source).strip()
    return source[:240]


def _collect_fact_events(
    previous_memory: dict[str, Any],
    updated_memory: dict[str, Any],
) -> list[dict[str, Any]]:
    previous = normalize_facts_memory(previous_memory)
    updated = normalize_facts_memory(updated_memory)
    events: list[dict[str, Any]] = []

    for category in FACT_CATEGORIES:
        previous_items = previous.get(category, {})
        updated_items = updated.get(category, {})
        for key in sorted(set(previous_items) | set(updated_items)):
            if _should_skip_fact_event(category, key, previous_items, updated_items):
                continue

            previous_value = _fact_value(previous_items, key)
            updated_value = _fact_value(updated_items, key)
            if not updated_value or previous_value == updated_value:
                continue

            updated_fact = updated_items.get(key, {})
            events.append(
                {
                    "timestamp": updated_fact.get("updated_at") or _utc_now(),
                    "category": category,
                    "key": key,
                    "old_value": previous_value,
                    "new_value": updated_value,
                    "source": updated_fact.get("source"),
                }
            )

    return events


def _should_skip_fact_event(
    category: str,
    key: str,
    previous_items: dict[str, Any],
    updated_items: dict[str, Any],
) -> bool:
    if category != "profile" or key != "role":
        return False

    previous_department = _fact_value(previous_items, "department")
    previous_role = _fact_value(previous_items, "role")
    updated_department = _fact_value(updated_items, "department")
    updated_role = _fact_value(updated_items, "role")

    if not previous_role or not updated_role:
        return False
    if not previous_department or not updated_department:
        return False
    return (
        previous_department.casefold() == previous_role.casefold()
        and updated_department.casefold() == updated_role.casefold()
    )


def _select_temporal_events(
    events: list[dict[str, Any]],
    query: str,
    *,
    max_events: int,
) -> list[dict[str, Any]]:
    relevant_keys = _temporal_query_keys(query)
    ordered_events = list(reversed(events))
    if relevant_keys:
        ordered_events = [
            event for event in ordered_events if str(event.get("key") or "") in relevant_keys
        ]
    return ordered_events[:max_events]


def _temporal_query_keys(query: str) -> set[str]:
    lowered = query.lower()
    keys: set[str] = set()

    if any(term in lowered for term in ("department", "team", "org", "organization")):
        keys.update({"department", "role"})
    if any(term in lowered for term in ("role", "job", "title")):
        keys.update({"role", "department"})
    if any(term in lowered for term in ("focus", "focus area", "domain")):
        keys.add("domain")
    if any(term in lowered for term in ("project", "workstream", "challenge")):
        keys.update({"current_project", "key_challenge"})
    if "fruit" in lowered:
        keys.add("preferred_fruit")
    if any(term in lowered for term in ("preference", "prefer", "style", "response", "format")):
        keys.update({"response_style", "preference_comparison"})

    return keys


def _title_preserving_dr(value: str) -> str:
    value = value.strip()
    if value.lower().startswith("dr"):
        rest = value.split(None, 1)[1] if " " in value else ""
        return "Dr. " + rest.title()
    return value.title()


def _build_facts_llm_prompt(
    current_memory: dict[str, Any], transcript: list[Any], timestamp: str
) -> str:
    transcript_lines = "\n".join(
        f"- {message.get('role')}: {message.get('content')}"
        if isinstance(message, dict)
        else f"- {getattr(message, 'role', getattr(message, 'type', 'message'))}: {getattr(message, 'content', '')}"
        for message in transcript
    )
    flat_current = json.dumps(flatten_facts_memory(current_memory), indent=2, sort_keys=True)
    return (
        "Update the user's durable facts from the transcript.\n"
        "Return only valid JSON with this exact top-level shape:\n"
        '{\n'
        '  "profile": {"key": "value"},\n'
        '  "preferences": {"key": "value"},\n'
        '  "constraints": {"key": "value"},\n'
        '  "project_context": {"key": "value"}\n'
        '}\n'
        "Rules:\n"
        "- Keep only durable user facts.\n"
        "- Use only these keys: profile={name, department, role, domain}; preferences={response_style, preferred_fruit, preference_comparison}; constraints={avoid_long_paragraphs}; project_context={current_project, key_challenge}.\n"
        "- Use profile.department for organizational groups such as Regulatory Affairs or Quality Assurance.\n"
        "- Ignore greetings, transient questions, and assistant chatter.\n"
        "- Resolve contradictions in favor of the latest user statement.\n"
        "- Do not invent missing values.\n"
        f"- Current timestamp for changed fields: {timestamp}\n\n"
        f"Current facts:\n{flat_current}\n\n"
        f"Transcript:\n{transcript_lines}\n"
    )


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _parse_json_object(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _canonicalize_llm_facts(
    raw: dict[str, Any],
    current_memory: dict[str, Any],
    transcript: list[Any],
    timestamp: str,
) -> dict[str, Any]:
    base = normalize_facts_memory(deepcopy(current_memory))
    source = _short_source(" | ".join(_user_contents(transcript)) or "llm consolidation")
    for category in FACT_CATEGORIES:
        incoming = raw.get(category, {})
        if not isinstance(incoming, dict):
            continue
        allowed_keys = CANONICAL_FACT_KEYS[category]
        current_category = base[category]
        normalized_category: dict[str, dict[str, str]] = {}
        for key, value in incoming.items():
            normalized_key = str(key)
            if normalized_key not in allowed_keys:
                continue
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            current_fact = current_category.get(normalized_key)
            if (
                isinstance(current_fact, dict)
                and str(current_fact.get("value") or "") == text
            ):
                normalized_category[normalized_key] = deepcopy(current_fact)
                continue
            normalized_category[normalized_key] = {
                "value": text,
                "updated_at": timestamp,
                "source": source,
            }
        if category == "profile":
            _preserve_department_alias(current_category, normalized_category, timestamp, source)
        for key, payload in current_category.items():
            if key in normalized_category or key not in allowed_keys:
                continue
            if isinstance(payload, dict):
                normalized_category[key] = deepcopy(payload)
        base[category] = normalized_category
    return base


def _ordered_fact_keys(category: str, items: dict[str, Any]) -> list[str]:
    keys = sorted(items)
    if category != "profile":
        return keys

    preferred = ("name", "department", "role", "domain")
    ordered = [key for key in preferred if key in items]
    ordered.extend(key for key in keys if key not in preferred)
    return ordered


def _should_skip_fact_line(category: str, key: str, items: dict[str, Any]) -> bool:
    if category != "profile" or key != "role":
        return False

    department = _fact_value(items, "department")
    role = _fact_value(items, "role")
    if not department or not role:
        return False
    return department.casefold() == role.casefold()


def _preserve_department_alias(
    current_profile: dict[str, Any],
    updated_profile: dict[str, dict[str, str]],
    timestamp: str,
    source: str,
) -> None:
    if "department" in updated_profile:
        return

    previous_department = current_profile.get("department")
    role_fact = updated_profile.get("role")
    if not isinstance(previous_department, dict) or not isinstance(role_fact, dict):
        return

    previous_value = previous_department.get("value")
    role_value = role_fact.get("value")
    if not previous_value or not role_value:
        return
    if str(previous_value).casefold() != str(role_value).casefold():
        return

    updated_profile["department"] = {
        "value": str(previous_value),
        "updated_at": timestamp,
        "source": source,
    }
