import argparse

from dotenv import load_dotenv

from agent.config import load_config
from agent.core import make_agent
from agent.search.service import QmsSearchService


def _print_search_result(result: dict[str, object]) -> None:
    print(f"\nAssistant: {result.get('answer', '')}\n")
    citations = result.get("citations", [])
    if isinstance(citations, list) and citations:
        print("Citations:")
        for citation in citations[:8]:
            if not isinstance(citation, dict):
                continue
            label = (
                f"{citation.get('doc_id')} Rev {citation.get('revision')}"
                f" - {citation.get('section')}"
            )
            print(f"- {label}")
        print()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="CLI Chat Agent")
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5-20251001",
        help="Model string, e.g. openai:gpt-4o, anthropic:claude-haiku-4-5-20251001, google_genai:gemini-2.5-flash",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Custom system prompt",
    )
    parser.add_argument(
        "--qms-search",
        action="store_true",
        help="Answer turns with citation-backed MedAI QMS search instead of the generic chat agent.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "local", "hybrid", "hosted"],
        help="QMS search mode when --qms-search is enabled.",
    )
    parser.add_argument(
        "--limit",
        default=16,
        type=int,
        help="Maximum QMS search hits when --qms-search is enabled.",
    )
    args = parser.parse_args()

    search_service = None
    agent = None
    if args.qms_search:
        config = load_config()
        search_service = QmsSearchService(
            config, use_hash_embeddings=config.use_hash_embeddings
        )
    else:
        agent = make_agent(args.model, args.system)
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
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})
        if search_service is not None:
            result = search_service.search(user_input, mode=args.mode, limit=args.limit)
            _print_search_result(result)
            messages.append({"role": "assistant", "content": str(result.get("answer", ""))})
        else:
            result = agent.invoke({"messages": messages})
            ai_msg = result["messages"][-1]
            print(f"\nAssistant: {ai_msg.content}\n")
            messages = result["messages"]


if __name__ == "__main__":
    main()
