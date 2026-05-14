import argparse
from contextlib import nullcontext
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from agent.config import load_config
from agent.core import make_agent
from agent.search.service import QmsSearchService


PROMPT_PREFIX_RE = re.compile(r"^\s*(you|user|q|query)\s*:\s*", re.IGNORECASE)


class CliRenderer:
    def __init__(
        self,
        *,
        pretty: bool = False,
        progress: bool = False,
        stream=None,
    ):
        self.pretty = pretty
        self.progress = progress
        self.console = Console(
            file=stream or sys.stdout,
            force_terminal=False,
            color_system=None,
            width=120,
        )

    def banner(self) -> None:
        self.console.print("Chat started. Type 'quit' to exit.\n")

    def render_progress(self, message: str):
        if self.pretty and self.progress:
            return self.console.status(message, spinner="dots")
        return nullcontext()

    def render_generic_answer(self, content: str) -> None:
        if not self.pretty:
            self.console.print(f"\nAssistant: {content}\n")
            return
        self.console.print(Panel(str(content), title="Assistant", box=box.ASCII))
        self.console.print()

    def render_qms_result(
        self,
        result: dict[str, object],
        *,
        trace: bool = False,
        full_citations: bool = False,
        raw_trace: bool = False,
    ) -> None:
        if not self.pretty:
            _print_search_result(
                result,
                trace=trace,
                full_citations=full_citations,
            )
            return

        subtitle = _summary_subtitle(result)
        self.console.print(
            Panel(
                str(result.get("answer", "")),
                title="Assistant",
                subtitle=subtitle,
                box=box.ASCII,
            )
        )
        self._render_summary(result)
        self._render_warnings(result)
        self._render_citations(result, full_citations=full_citations)
        if trace:
            if raw_trace:
                self.console.print("Trace:")
                self.console.print(json.dumps(result.get("debug_trace", {}), indent=2))
                self.console.print()
            else:
                self._render_trace(result.get("debug_trace", {}))
        self._render_repeated_answer(result)

    def _render_repeated_answer(self, result: dict[str, object]) -> None:
        self.console.print(
            Panel(
                str(result.get("answer", "")),
                title="Assistant (Repeated)",
                box=box.ASCII,
            )
        )
        self.console.print()

    def _render_summary(self, result: dict[str, object]) -> None:
        table = Table(title="Run Summary", box=box.SIMPLE, show_header=True)
        table.add_column("Field")
        table.add_column("Value")
        citations = result.get("citations", [])
        retrieved = result.get("retrieved_documents", [])
        warnings = result.get("warnings", [])
        table.add_row("Mode", str(result.get("mode") or "unknown"))
        table.add_row("Retrieval backend", str(result.get("retrieval_backend") or "unknown"))
        table.add_row("Citations", str(len(citations) if isinstance(citations, list) else 0))
        table.add_row(
            "Retrieved documents",
            str(len(retrieved) if isinstance(retrieved, list) else 0),
        )
        table.add_row("Warnings", str(len(warnings) if isinstance(warnings, list) else 0))
        self.console.print(table)

    def _render_warnings(self, result: dict[str, object]) -> None:
        warnings = result.get("warnings", [])
        if not isinstance(warnings, list) or not warnings:
            return
        table = Table(title="Warnings", box=box.SIMPLE, show_header=False)
        table.add_column("Warning")
        for warning in warnings[:8]:
            table.add_row(str(warning))
        if len(warnings) > 8:
            table.add_row(f"... {len(warnings) - 8} more warning(s) not shown")
        self.console.print(table)

    def _render_citations(
        self,
        result: dict[str, object],
        *,
        full_citations: bool,
    ) -> None:
        citations = result.get("citations", [])
        if not isinstance(citations, list) or not citations:
            return
        for index, citation in enumerate(citations[:8], start=1):
            if not isinstance(citation, dict):
                continue
            table = Table.grid(padding=(0, 1))
            table.add_column(justify="right")
            table.add_column()
            table.add_row("Document", _citation_doc_label(citation))
            table.add_row("Title", str(citation.get("title") or "unknown"))
            table.add_row("Section", str(citation.get("section") or "unknown"))
            table.add_row("Evidence type", _citation_evidence_label(citation))
            table.add_row("Filename", str(citation.get("filename") or "unknown"))
            if citation.get("chunk_id"):
                table.add_row("Chunk", str(citation["chunk_id"]))
            if citation.get("score") is not None:
                table.add_row("Score", str(citation["score"]))
            if full_citations:
                for path_label, path_value in _citation_paths(citation):
                    table.add_row(path_label, path_value)
            self.console.print(Panel(table, title=f"Source {index}", box=box.ASCII))
        if len(citations) > 8:
            self.console.print(f"... {len(citations) - 8} additional citation(s) not shown.")

    def _render_trace(self, trace: object) -> None:
        if not isinstance(trace, dict):
            self.console.print(Panel(str(trace), title="Operational Trace", box=box.ASCII))
            return
        service_trace = trace.get("service_trace", {})
        panels = [
            Panel(_trace_table(_input_trace_rows(trace)), title="Input", box=box.ASCII),
            Panel(_trace_table(_query_trace_rows(trace)), title="Query", box=box.ASCII),
            Panel(
                _trace_table(_routing_trace_rows(trace, service_trace)),
                title="Routing",
                box=box.ASCII,
            ),
            Panel(
                _trace_table(_result_trace_rows(trace, service_trace)),
                title="Results",
                box=box.ASCII,
            ),
        ]
        if isinstance(service_trace, dict) and service_trace:
            panels.append(
                Panel(
                    json.dumps(service_trace, indent=2),
                    title="Service trace",
                    box=box.ASCII,
                )
            )
        self.console.print(Panel(Group(*panels), title="Operational Trace", box=box.ASCII))


def _print_search_result(
    result: dict[str, object],
    *,
    trace: bool = False,
    full_citations: bool = False,
) -> None:
    print(f"\nAssistant: {result.get('answer', '')}\n")
    if result.get("retrieval_backend"):
        print(f"Retrieval backend: {result['retrieval_backend']}")
    warnings = result.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        print("Warnings:")
        for warning in warnings[:8]:
            print(f"- {warning}")
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
            if full_citations:
                for path_label, path_value in _citation_paths(citation):
                    print(f"  {path_label}: {path_value}")
        print()
    if trace:
        print("Trace:")
        print(json.dumps(result.get("debug_trace", {}), indent=2))
        print()
    print(f"Assistant (repeated): {result.get('answer', '')}\n")


def _summary_subtitle(result: dict[str, object]) -> str:
    mode = result.get("mode") or "unknown"
    backend = result.get("retrieval_backend") or "unknown"
    return f"mode={mode} | backend={backend}"


def _citation_doc_label(citation: dict[str, object]) -> str:
    doc_id = str(citation.get("doc_id") or "unknown")
    revision = citation.get("revision")
    if revision:
        return f"{doc_id} Rev {revision}"
    return doc_id


def _trace_table(rows: list[tuple[str, object]]) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right")
    table.add_column()
    for label, value in rows:
        if isinstance(value, (list, dict)):
            value = json.dumps(value, indent=2)
        table.add_row(label, str(value))
    return table


def _input_trace_rows(trace: dict[str, object]) -> list[tuple[str, object]]:
    return [
        ("raw input", trace.get("raw_input", "")),
        ("normalized input", trace.get("normalized_input", "")),
        ("prefix stripped", trace.get("input_prefix_stripped", False)),
        ("stripped prefixes", trace.get("stripped_prefixes", [])),
        ("follow-up", trace.get("follow_up_detected", False)),
    ]


def _query_trace_rows(trace: dict[str, object]) -> list[tuple[str, object]]:
    return [
        ("original query", trace.get("original_query", "")),
        ("resolved query", trace.get("resolved_query", "")),
        ("context doc", trace.get("context_doc_id", "")),
        ("context revision", trace.get("context_revision", "")),
        ("context title", trace.get("context_title", "")),
    ]


def _routing_trace_rows(
    trace: dict[str, object],
    service_trace: object,
) -> list[tuple[str, object]]:
    service = service_trace if isinstance(service_trace, dict) else {}
    return [
        ("requested mode", trace.get("requested_mode", "")),
        ("retrieval backend", trace.get("retrieval_backend", "")),
        ("category", service.get("query_plan_category", "")),
        ("strategy", service.get("query_plan_strategy", "")),
    ]


def _result_trace_rows(
    trace: dict[str, object],
    service_trace: object,
) -> list[tuple[str, object]]:
    service = service_trace if isinstance(service_trace, dict) else {}
    return [
        ("retrieved count", service.get("retrieved_count", "")),
        ("citation count", service.get("citation_count", "")),
        ("warnings", trace.get("warnings", [])),
    ]


def _citation_paths(citation: dict[str, object]) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    markdown_path = citation.get("markdown_path_abs") or _absolute_path(
        citation.get("markdown_path")
    )
    source_path = citation.get("source_path_abs") or _absolute_path(
        citation.get("source_path")
    )
    if markdown_path:
        paths.append(("Markdown path", str(markdown_path)))
    if source_path:
        paths.append(("Source path", str(source_path)))
    return paths


def _citation_evidence_label(citation: dict[str, object]) -> str:
    evidence_type = str(citation.get("evidence_type") or "metadata")
    support = str(citation.get("support_level") or "document")
    table_index = citation.get("table_index")
    row_start = citation.get("row_start")
    row_end = citation.get("row_end")
    parts = [f"{evidence_type} / {support}"]
    if table_index not in (None, ""):
        parts.append(f"Table {table_index}")
    if row_start not in (None, ""):
        row = f"Row {row_start}"
        if row_end not in (None, "", row_start):
            row += f"-{row_end}"
        parts.append(row)
    return " -> ".join(parts)


def _absolute_path(value: object) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve(strict=False))


def _resolve_qms_followup(
    user_input: str,
    qms_turns: list[dict[str, object]],
) -> tuple[str, dict[str, object]]:
    trace: dict[str, object] = {
        "original_query": user_input,
        "resolved_query": user_input,
        "follow_up_detected": False,
    }
    if not qms_turns:
        return user_input, trace
    context = _last_context_citation(qms_turns)
    if context is None or not _looks_like_followup(user_input):
        return user_input, trace

    doc_id = str(context.get("doc_id") or "")
    revision = str(context.get("revision") or "")
    title = str(context.get("title") or "")
    label = " ".join(part for part in [doc_id, f"Rev {revision}" if revision else ""] if part)
    if not label:
        return user_input, trace
    resolved = f"{user_input} Previous cited document: {label}"
    if title:
        resolved += f" - {title}"
    resolved += "."
    trace.update(
        {
            "resolved_query": resolved,
            "follow_up_detected": True,
            "context_doc_id": doc_id,
            "context_revision": revision,
            "context_title": title,
        }
    )
    return resolved, trace


def _normalize_cli_input(raw: str) -> tuple[str, dict[str, object]]:
    cleaned = raw.strip()
    stripped_prefixes: list[str] = []
    while True:
        match = PROMPT_PREFIX_RE.match(cleaned)
        if match is None:
            break
        stripped_prefixes.append(match.group(1))
        cleaned = cleaned[match.end() :].strip()
    return cleaned, {
        "raw_input": raw,
        "normalized_input": cleaned,
        "input_prefix_stripped": bool(stripped_prefixes),
        "stripped_prefixes": stripped_prefixes,
    }


def _qms_prompt(args) -> str:
    if getattr(args, "qms_search", False) and getattr(args, "json", False):
        return ""
    if getattr(args, "qms_search", False):
        return "QMS> "
    return "You: "


def _stdout_is_tty() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _should_use_progress(args, *, is_tty: bool | None = None) -> bool:
    if is_tty is None:
        is_tty = _stdout_is_tty()
    return bool(
        is_tty
        and not getattr(args, "json", False)
        and not getattr(args, "plain", False)
        and not getattr(args, "no_progress", False)
    )


def _should_use_pretty(args, *, is_tty: bool | None = None) -> bool:
    if is_tty is None:
        is_tty = _stdout_is_tty()
    return bool(is_tty and not getattr(args, "json", False) and not getattr(args, "plain", False))


def _apply_cli_answer_overrides(
    result: dict[str, object],
    *,
    user_input: str,
    qms_turns: list[dict[str, object]],
) -> None:
    if not _is_full_path_request(user_input):
        return
    citation = _first_path_citation(result) or _last_context_citation(qms_turns)
    if citation is None:
        return
    doc_id = str(citation.get("doc_id") or "").strip()
    revision = str(citation.get("revision") or "").strip()
    label = " ".join(part for part in [doc_id, f"Rev {revision}" if revision else ""] if part)
    if not label:
        label = "the cited source"
    result["answer"] = f"Full citation paths are listed below for {label}."
    trace = result.get("debug_trace")
    if isinstance(trace, dict):
        trace["cli_answer_override"] = "full_path_citation"


def _is_full_path_request(query: str) -> bool:
    lower = query.lower()
    return any(term in lower for term in ["full path", "pathname", "full pathname"])


def _first_path_citation(result: dict[str, object]) -> dict[str, object] | None:
    citations = result.get("citations", [])
    if not isinstance(citations, list):
        return None
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        if citation.get("markdown_path_abs") or citation.get("source_path_abs"):
            return citation
    return None


def _last_context_citation(qms_turns: list[dict[str, object]]) -> dict[str, object] | None:
    for turn in reversed(qms_turns):
        result = turn.get("result")
        if not isinstance(result, dict):
            continue
        citations = result.get("citations", [])
        if isinstance(citations, list):
            for citation in citations:
                if isinstance(citation, dict) and citation.get("doc_id"):
                    return citation
    return None


def _looks_like_followup(query: str) -> bool:
    lower = query.lower()
    followup_patterns = [
        r"\b(that|this|same)\s+(document|doc|record|source|citation|revision|version)\b",
        r"\b(it|that|this)\?\s*$",
        r"\bwhat\s+revision\s+is\s+(that|it|this)\b",
        r"\bwhich\s+document\b",
        r"\bfull\s+path(?:name)?\b",
        r"\bpath(?:name)?\b",
        r"\bcitation\b",
        r"\bcompare\s+(this|that|same|it)\b",
        r"\bwhat\s+changed\b",
        r"\blatest\s+version\b",
        r"\bprevious\s+versions?\b",
        r"\brev\s+[a-z0-9]\b",
    ]
    return any(re.search(pattern, lower) for pattern in followup_patterns)


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
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print QMS operational traces when --qms-search is enabled.",
    )
    parser.add_argument(
        "--full-citations",
        action="store_true",
        help="Print full local source paths for QMS citations.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print each QMS turn as structured JSON.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable Rich terminal formatting and use stable plain text output.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the interactive progress spinner while preserving pretty output.",
    )
    parser.add_argument(
        "--raw-trace",
        action="store_true",
        help="Print raw JSON trace output instead of formatted trace panels.",
    )
    parser.add_argument(
        "--no-followup",
        action="store_true",
        help="Disable QMS follow-up context carryover for every turn.",
    )
    parser.add_argument(
        "--force-strategy",
        choices=[
            "hybrid",
            "exact_then_hybrid",
            "sql_count",
            "sql_list",
            "revision_chain",
            "revision_diff",
            "multi_hop",
        ],
        default=None,
        help="Debug only: override the QMS query planner strategy.",
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
    qms_turns: list[dict[str, object]] = []

    json_qms_output = bool(args.qms_search and args.json)
    renderer = CliRenderer(
        pretty=_should_use_pretty(args),
        progress=_should_use_progress(args),
    )
    prompt = _qms_prompt(args)
    if not json_qms_output:
        renderer.banner()

    while True:
        try:
            raw_user_input = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_input, input_trace = _normalize_cli_input(raw_user_input)
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() in ("/clear", "/reset"):
            messages.clear()
            qms_turns.clear()
            if args.json and args.qms_search:
                print(json.dumps({"event": "context_cleared"}))
            else:
                renderer.render_generic_answer("Session context cleared.")
            continue

        messages.append({"role": "user", "content": user_input})
        if search_service is not None:
            with renderer.render_progress("Resolving follow-up context"):
                if args.no_followup:
                    resolved_query = user_input
                    trace = {
                        "original_query": user_input,
                        "resolved_query": user_input,
                        "follow_up_detected": False,
                        "follow_up_disabled": True,
                    }
                else:
                    resolved_query, trace = _resolve_qms_followup(user_input, qms_turns)
            with renderer.render_progress("Searching MedAI QMS"):
                result = search_service.search(
                    resolved_query,
                    mode=args.mode,
                    limit=args.limit,
                    force_strategy=args.force_strategy,
                )
            service_trace = result.get("debug_trace", {})
            result["debug_trace"] = {
                **trace,
                **input_trace,
                "service_trace": service_trace,
                "requested_mode": result.get("mode"),
                "retrieval_backend": result.get("retrieval_backend"),
                "warnings": result.get("warnings", []),
            }
            with renderer.render_progress("Validating citations"):
                _apply_cli_answer_overrides(
                    result,
                    user_input=user_input,
                    qms_turns=qms_turns,
                )
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                with renderer.render_progress("Rendering answer"):
                    renderer.render_qms_result(
                        result,
                        trace=args.trace,
                        full_citations=args.full_citations,
                        raw_trace=args.raw_trace,
                    )
            qms_turns.append(
                {
                    "user_input": user_input,
                    "resolved_query": resolved_query,
                    "result": result,
                }
            )
            messages.append({"role": "assistant", "content": str(result.get("answer", ""))})
        else:
            with renderer.render_progress("Sending message"):
                pass
            with renderer.render_progress("Waiting for model response"):
                result = agent.invoke({"messages": messages})
            ai_msg = result["messages"][-1]
            with renderer.render_progress("Rendering answer"):
                renderer.render_generic_answer(str(ai_msg.content))
            messages = result["messages"]


if __name__ == "__main__":
    main()
