from __future__ import annotations

import os
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from agent.core import DEFAULT_MODEL_STR, make_agent
from agent.memory import (
    DEFAULT_FACTS_MEMORY,
    DEFAULT_SUMMARY_MEMORY,
    append_facts_events,
    append_summary_facts_events,
    build_summary_memory_from_facts,
    delete_summary_facts_events,
    delete_summary_facts_memory,
    delete_facts_events,
    facts_path,
    facts_events_path,
    llm_consolidate_facts_memory,
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
    summary_path,
    consolidate_facts_memory,
)


DEFAULT_CHAT_MODEL = DEFAULT_MODEL_STR
DEFAULT_EXTRACTOR_MODEL = "anthropic:claude-haiku-4-5"
DEFAULT_FACTS_EXTRACTOR = "hybrid"
MEMORY_TYPES = ("none", "facts", "summary")
FACTS_EXTRACTORS = ("deterministic", "hybrid", "llm")

AgentFactory = Callable[[str, str | None], Any]

DEMO_USER_DEFAULT = "blank_demo"
DEMO_USERS: dict[str, dict[str, Any]] = {
    "blank_demo": {
        "key": "blank_demo",
        "label": "Blank demo user",
        "user_id": "demo_user",
        "description": "Fresh start with empty facts and summary memory.",
        "best_for": "Manual freeform testing from a clean state.",
        "facts": deepcopy(DEFAULT_FACTS_MEMORY),
        "summary": deepcopy(DEFAULT_SUMMARY_MEMORY),
    },
    "regulatory_lead": {
        "key": "regulatory_lead",
        "label": "Regulatory lead",
        "user_id": "demo_regulatory_lead",
        "description": "Preloaded regulatory affairs profile with current project context.",
        "best_for": "Identity recall and project context recall.",
        "facts": {
            "profile": {
                "name": {
                    "value": "Dr. Sarah Chen",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                },
                "department": {
                    "value": "Regulatory Affairs",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                },
                "role": {
                    "value": "Regulatory Affairs",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                },
                "domain": {
                    "value": "cardiac 510(k) submissions",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                },
            },
            "preferences": {
                "response_style": {
                    "value": "concise bullet-point answers",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                }
            },
            "constraints": {
                "avoid_long_paragraphs": {
                    "value": "no long paragraphs",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                }
            },
            "project_context": {
                "current_project": {
                    "value": "new catheter 510(k)",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                },
                "key_challenge": {
                    "value": "choosing between two predicate devices",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                },
            },
        },
        "summary": {
            "summary": (
                "User is Dr. Sarah Chen in Regulatory Affairs focused on cardiac 510(k) "
                "submissions. They prefer concise bullet-point answers, avoid long "
                "paragraphs, and are currently working on a new catheter 510(k) with a "
                "predicate-device selection challenge."
            ),
            "updated_at": "2026-04-10T12:00:00Z",
        },
    },
    "personal_preferences": {
        "key": "personal_preferences",
        "label": "Personal preferences",
        "user_id": "demo_personal_preferences",
        "description": "Simple personal identity and preference memory for plain-language demos.",
        "best_for": "Personal preference recall and same-chat vs restart walkthroughs.",
        "facts": {
            "profile": {
                "name": {
                    "value": "Andrew",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                }
            },
            "preferences": {
                "preferred_fruit": {
                    "value": "mango",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                },
                "preference_comparison": {
                    "value": "chocolate over peanut butter",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                },
            },
            "constraints": {},
            "project_context": {},
        },
        "summary": {
            "summary": (
                "User is Andrew. Their preferred fruit is mango. They prefer chocolate "
                "over peanut butter."
            ),
            "updated_at": "2026-04-10T12:00:00Z",
        },
    },
    "style_constrained": {
        "key": "style_constrained",
        "label": "Style constrained",
        "user_id": "demo_style_constrained",
        "description": "Quality Assurance profile with strong answer-format constraints.",
        "best_for": "Preference application and contradiction-update demos.",
        "facts": {
            "profile": {
                "name": {
                    "value": "Jordan Lee",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                },
                "department": {
                    "value": "Quality Assurance",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                },
                "role": {
                    "value": "Quality Assurance",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                },
            },
            "preferences": {
                "response_style": {
                    "value": "concise bullet-point answers",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                }
            },
            "constraints": {
                "avoid_long_paragraphs": {
                    "value": "no long paragraphs",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                }
            },
            "project_context": {
                "current_project": {
                    "value": "CAPA audit preparation",
                    "updated_at": "2026-04-10T12:00:00Z",
                    "source": "Seeded demo user",
                }
            },
        },
        "summary": {
            "summary": (
                "User is Jordan Lee in Quality Assurance. They prefer concise "
                "bullet-point answers, avoid long paragraphs, and are working on CAPA "
                "audit preparation."
            ),
            "updated_at": "2026-04-10T12:00:00Z",
        },
    },
}


def compose_system_prompt(base_prompt: str | None, memory_block: str) -> str | None:
    memory_block = memory_block.strip()
    if memory_block:
        memory_block = (
            f"{memory_block}\n\n"
            "Use this long-term memory when relevant. Follow durable preferences and "
            "constraints unless the current user request explicitly overrides them."
        )
    if base_prompt and memory_block:
        return f"{base_prompt.strip()}\n\n{memory_block}"
    if base_prompt:
        return base_prompt
    if memory_block:
        return memory_block
    return None


def build_system_prompt_preview(
    *,
    memory_type: str,
    memory_root: Path | str,
    user_id: str,
    base_system_prompt: str | None = None,
) -> str | None:
    current_memory, _ = read_memory_snapshot(memory_type, memory_root, user_id)
    memory_block = _render_memory_block(memory_type, current_memory)
    return compose_system_prompt(base_system_prompt, memory_block)


def read_memory_snapshot(
    memory_type: str, memory_root: Path | str, user_id: str
) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(memory_root)
    if memory_type == "facts":
        return load_facts_memory(root, user_id), facts_path(root, user_id)
    if memory_type == "summary":
        return load_summary_memory(root, user_id), summary_path(root, user_id)
    return None, None


def read_facts_events_snapshot(
    memory_type: str, memory_root: Path | str, user_id: str
) -> tuple[list[dict[str, Any]] | None, Path | None]:
    root = Path(memory_root)
    if memory_type == "facts":
        return load_facts_events(root, user_id), facts_events_path(root, user_id)
    return None, None


def delete_memory_snapshot(
    memory_type: str, memory_root: Path | str, user_id: str
) -> tuple[dict[str, Any] | None, Path | None]:
    memory, path = read_memory_snapshot(memory_type, memory_root, user_id)
    if path and path.exists():
        path.unlink()
    if memory_type == "facts":
        delete_facts_events(memory_root, user_id)
    if memory_type == "summary":
        delete_summary_facts_memory(memory_root, user_id)
        delete_summary_facts_events(memory_root, user_id)
    return read_memory_snapshot(memory_type, memory_root, user_id)


def get_backend_health(memory_root: Path | str) -> dict[str, Any]:
    root = Path(memory_root)
    root.mkdir(parents=True, exist_ok=True)

    providers = {
        "anthropic": _provider_status("ANTHROPIC_API_KEY"),
        "openai": _provider_status("OPENAI_API_KEY"),
        "google": _provider_status("GOOGLE_API_KEY"),
    }
    chat_provider = _provider_for_model(DEFAULT_CHAT_MODEL)
    extractor_provider = _provider_for_model(DEFAULT_EXTRACTOR_MODEL)

    return {
        "status": "ok",
        "serverReachable": True,
        "memoryStore": {
            "path": str(root),
            "writable": os.access(root, os.W_OK),
        },
        "defaultChatModel": DEFAULT_CHAT_MODEL,
        "defaultExtractorModel": DEFAULT_EXTRACTOR_MODEL,
        "providers": providers,
        "chatModel": {
            "provider": chat_provider,
            "configured": providers.get(chat_provider, {}).get("configured", False),
        },
        "extractorModel": {
            "provider": extractor_provider,
            "configured": providers.get(extractor_provider, {}).get("configured", False),
        },
        "note": (
            "These checks confirm that the API responded, the memory store is writable, "
            "and provider environment variables are present. They do not validate API "
            "keys with the upstream model providers."
        ),
    }


def list_demo_users() -> list[dict[str, str]]:
    return [
        {
            "key": demo["key"],
            "label": demo["label"],
            "userId": demo["user_id"],
            "description": demo["description"],
            "bestFor": demo["best_for"],
        }
        for demo in DEMO_USERS.values()
    ]


def load_demo_user(memory_root: Path | str, demo_user_key: str) -> dict[str, Any]:
    if demo_user_key not in DEMO_USERS:
        raise KeyError(f"Unknown demo user: {demo_user_key}")

    demo = DEMO_USERS[demo_user_key]
    user_id = demo["user_id"]
    root = Path(memory_root)

    facts = normalize_facts_for_save(demo["facts"])
    summary = normalize_summary_for_save(demo["summary"])

    delete_facts_events(root, user_id)
    delete_summary_facts_events(root, user_id)
    save_facts_memory(root, user_id, facts)
    save_summary_facts_memory(root, user_id, facts)
    save_summary_memory(root, user_id, summary)

    return {
        "demoUser": {
            "key": demo["key"],
            "label": demo["label"],
            "userId": user_id,
            "description": demo["description"],
            "bestFor": demo["best_for"],
        },
        "userId": user_id,
        "facts": load_facts_memory(root, user_id),
        "factsEvents": load_facts_events(root, user_id),
        "factsEventsPath": str(facts_events_path(root, user_id)),
        "summary": load_summary_memory(root, user_id),
        "promptPreviews": {
            "none": None,
            "facts": build_system_prompt_preview(
                memory_type="facts",
                memory_root=root,
                user_id=user_id,
            ),
            "summary": build_system_prompt_preview(
                memory_type="summary",
                memory_root=root,
                user_id=user_id,
            ),
        },
    }


def run_chat_turn(
    *,
    messages: list[dict[str, Any]],
    memory_type: str,
    user_id: str,
    memory_root: Path | str,
    model: str = DEFAULT_CHAT_MODEL,
    base_system_prompt: str | None = None,
    facts_extractor: str = DEFAULT_FACTS_EXTRACTOR,
    extractor_model: str = DEFAULT_EXTRACTOR_MODEL,
    agent_factory: AgentFactory = make_agent,
) -> dict[str, Any]:
    current_memory, memory_path = read_memory_snapshot(memory_type, memory_root, user_id)
    memory_block = _render_memory_block(memory_type, current_memory)
    temporal_block = _render_temporal_memory_block(
        memory_type=memory_type,
        memory_root=memory_root,
        user_id=user_id,
        messages=messages,
    )
    combined_memory_block = _combine_memory_blocks(memory_block, temporal_block)
    system_prompt_preview = compose_system_prompt(base_system_prompt, combined_memory_block)
    agent = agent_factory(model, system_prompt_preview)
    result = agent.invoke({"messages": messages})
    updated_messages = result["messages"]
    assistant_message = updated_messages[-1]
    reply = getattr(assistant_message, "content", "")

    turn_delta = _latest_exchange(updated_messages)
    updated_memory = _persist_turn_memory(
        memory_type=memory_type,
        memory_root=memory_root,
        user_id=user_id,
        current_memory=current_memory,
        turn_delta=turn_delta,
        full_transcript=updated_messages,
        facts_extractor=facts_extractor,
        extractor_model=extractor_model,
    )

    return {
        "reply": reply,
        "messages": updated_messages,
        "memory": updated_memory,
        "memory_path": str(memory_path) if memory_path else None,
        "model_used": model,
        "system_prompt": system_prompt_preview,
    }


def finalize_chat_session(
    *,
    messages: list[Any],
    memory_type: str,
    user_id: str,
    memory_root: Path | str,
    facts_extractor: str = DEFAULT_FACTS_EXTRACTOR,
    extractor_model: str = DEFAULT_EXTRACTOR_MODEL,
) -> dict[str, Any] | None:
    if memory_type != "facts" or facts_extractor not in {"hybrid", "llm"} or not messages:
        memory, _ = read_memory_snapshot(memory_type, memory_root, user_id)
        return memory

    current_memory = load_facts_memory(memory_root, user_id)
    try:
        refined = llm_consolidate_facts_memory(
            current_memory,
            messages,
            extractor_model=extractor_model,
        )
    except Exception:
        return current_memory

    save_facts_memory(memory_root, user_id, refined)
    append_facts_events(memory_root, user_id, current_memory, refined)
    return refined


def _persist_turn_memory(
    *,
    memory_type: str,
    memory_root: Path | str,
    user_id: str,
    current_memory: dict[str, Any] | None,
    turn_delta: list[Any],
    full_transcript: list[Any],
    facts_extractor: str,
    extractor_model: str,
) -> dict[str, Any] | None:
    if memory_type == "none":
        return None

    if memory_type == "facts":
        current = current_memory or load_facts_memory(memory_root, user_id)
        if facts_extractor in {"deterministic", "hybrid"}:
            updated = consolidate_facts_memory(current, turn_delta)
        else:
            try:
                updated = llm_consolidate_facts_memory(
                    current,
                    full_transcript,
                    extractor_model=extractor_model,
                )
            except Exception:
                updated = current
        save_facts_memory(memory_root, user_id, updated)
        append_facts_events(memory_root, user_id, current, updated)
        return updated

    current = current_memory or load_summary_memory(memory_root, user_id)
    current_summary_facts = load_summary_facts_memory(memory_root, user_id)
    updated_summary_facts = consolidate_facts_memory(current_summary_facts, turn_delta)
    save_summary_facts_memory(memory_root, user_id, updated_summary_facts)
    append_summary_facts_events(
        memory_root,
        user_id,
        current_summary_facts,
        updated_summary_facts,
    )
    updated_summary = build_summary_memory_from_facts(
        updated_summary_facts,
        previous_summary=current,
    )
    save_summary_memory(memory_root, user_id, updated_summary)
    return updated_summary


def _render_memory_block(
    memory_type: str,
    memory: dict[str, Any] | None,
) -> str:
    if memory_type == "facts" and memory is not None:
        return render_facts_memory_block(memory)
    if memory_type == "summary" and memory is not None:
        return render_summary_memory_block(memory)
    return ""


def _render_temporal_memory_block(
    *,
    memory_type: str,
    memory_root: Path | str,
    user_id: str,
    messages: list[Any],
) -> str:
    if memory_type not in {"facts", "summary"}:
        return ""

    latest_user_message = _latest_user_message_content(messages)
    if not latest_user_message or not _is_temporal_query(latest_user_message):
        return ""

    if memory_type == "facts":
        events = load_facts_events(memory_root, user_id)
    else:
        events = load_summary_facts_events(memory_root, user_id)
    return render_temporal_facts_memory_block(events, latest_user_message)


def _combine_memory_blocks(*blocks: str) -> str:
    return "\n\n".join(block.strip() for block in blocks if block and block.strip())


def _latest_exchange(messages: list[Any]) -> list[Any]:
    latest_user_index = None
    for index in range(len(messages) - 1, -1, -1):
        role = _message_role(messages[index])
        if role in {"user", "human"}:
            latest_user_index = index
            break

    if latest_user_index is None:
        return messages[-1:]
    return messages[latest_user_index:]


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None) or getattr(message, "type", None)


def _latest_user_message_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        if _message_role(message) not in {"user", "human"}:
            continue
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(getattr(message, "content", "") or "")
    return ""


def _is_temporal_query(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "before",
            "previous",
            "previously",
            "prior",
            "earlier",
            "used to",
            "what changed",
            "history",
        )
    )


def _provider_for_model(model: str) -> str:
    return model.split(":", 1)[0] if ":" in model else "unknown"


def _provider_status(env_var: str) -> dict[str, Any]:
    return {
        "configured": bool(os.getenv(env_var)),
        "envVar": env_var,
    }


def normalize_facts_for_save(facts: dict[str, Any]) -> dict[str, Any]:
    return load_facts_memory_from_data(facts)


def normalize_summary_for_save(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": str(summary.get("summary") or ""),
        "updated_at": summary.get("updated_at") or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def load_facts_memory_from_data(data: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(DEFAULT_FACTS_MEMORY)
    for category in normalized:
        raw_category = data.get(category, {})
        if not isinstance(raw_category, dict):
            continue
        normalized[category] = deepcopy(raw_category)
    return normalized
