import argparse
from pathlib import Path

from dotenv import load_dotenv

from agent.chat_service import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EXTRACTOR_MODEL,
    FACTS_EXTRACTORS,
    MEMORY_TYPES,
    compose_system_prompt,
    finalize_chat_session,
    run_chat_turn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI Chat Agent")
    parser.add_argument(
        "--model",
        default=DEFAULT_CHAT_MODEL,
        help="Model string, e.g. openai:gpt-4o, anthropic:claude-haiku-4-5-20251001, google_genai:gemini-2.5-flash",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Custom system prompt",
    )
    parser.add_argument(
        "--memory-type",
        choices=MEMORY_TYPES,
        default="none",
        help="Long-term memory strategy to use across CLI sessions.",
    )
    parser.add_argument(
        "--user-id",
        default="demo_user",
        help="User identifier used to scope persisted memory.",
    )
    parser.add_argument(
        "--memory-root",
        default="memory_store",
        help="Directory where per-user memory JSON files are stored.",
    )
    parser.add_argument(
        "--facts-extractor",
        choices=FACTS_EXTRACTORS,
        default="hybrid",
        help="Facts extraction mode for long-term facts memory.",
    )
    parser.add_argument(
        "--extractor-model",
        default=DEFAULT_EXTRACTOR_MODEL,
        help="Model used for optional LLM-based facts consolidation.",
    )
    return parser


def main():
    """Run the interactive CLI loop and finalize session memory on exit."""
    load_dotenv()

    args = build_parser().parse_args()
    memory_root = Path(args.memory_root)
    messages = []

    print("Chat started. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        messages = [*messages, {"role": "user", "content": user_input}]
        result = run_chat_turn(
            messages=messages,
            memory_type=args.memory_type,
            user_id=args.user_id,
            memory_root=memory_root,
            model=args.model,
            base_system_prompt=args.system,
            facts_extractor=args.facts_extractor,
            extractor_model=args.extractor_model,
        )
        print(f"\nAssistant: {result['reply']}\n")
        messages = result["messages"]

    finalize_chat_session(
        messages=messages,
        memory_type=args.memory_type,
        user_id=args.user_id,
        memory_root=memory_root,
        facts_extractor=args.facts_extractor,
        extractor_model=args.extractor_model,
    )


if __name__ == "__main__":
    main()
