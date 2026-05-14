# CLI Guide

Run the agent as an interactive terminal chat. See the [core README](../README.md)
for setup, indexing, and eval commands.

## Generic Chat

```bash
# Default generic LangChain agent
uv run chat

# OpenAI
uv run chat --model openai:gpt-4o

# Google
uv run chat --model google_genai:gemini-2.5-flash

# Custom system prompt
uv run chat --system "You are a helpful coding assistant."
```

Generic chat prompts with `You: `. Type `quit` or `exit` to end the session.
The generic path uses `--model` and `--system`.

## QMS Search Chat

```bash
# Human-friendly QMS chat with pretty terminal output and progress on TTYs
uv run chat --qms-search --mode hybrid --limit 8

# Hosted-first auto mode with local fallback
uv run chat --qms-search --mode auto --limit 8

# Disable the dynamic spinner but keep pretty output on interactive terminals
uv run chat --qms-search --mode hybrid --limit 8 --no-progress

# Stable plain text output for copy/paste or CI logs
uv run chat --qms-search --mode hybrid --limit 8 --plain
```

QMS chat prompts with `QMS> `. It uses `QmsSearchService` and the configured
QMS models/indexes rather than the generic `--model` agent path.

The CLI strips pasted transcript prefixes before routing. Leading `You:`,
`User:`, `Q:`, and `Query:` labels are removed repeatedly and recorded in trace
metadata.

QMS chat keeps prior citation context for follow-ups. After finding `BOM-055 Rev
G`, prompts such as `What revision is that?`, `Show me the full pathname
citation.`, or `You: what changed in the latest version from previous versions?`
are resolved against the last cited document.

## Trace And Citation Output

```bash
# Formatted trace panels for human debugging
uv run chat --qms-search --mode hybrid --limit 8 --trace

# Raw JSON trace instead of formatted trace panels
uv run chat --qms-search --mode hybrid --limit 8 --trace --raw-trace

# Full absolute Markdown/source citation paths
uv run chat --qms-search --mode hybrid --limit 8 --full-citations

# Combined debugging mode
uv run chat --qms-search --mode hybrid --limit 8 --trace --full-citations
```

Formatted QMS output includes an answer panel, run summary, warnings, source
cards, grouped operational trace sections, and a repeated answer panel at the
bottom so long trace/citation runs do not require scrolling back to the top.
Citation cards include document ID, revision, title, section, filename, chunk
ID, score when present, and full paths when `--full-citations` is set.

Trace output is part of the query-path contract. It must preserve evidence
provenance and show raw/normalized input, planned intent, query-expansion
entities or terms, retrieval backend, warnings, references followed, and final
candidate/citation IDs. Query expansion may widen recall, but exact IDs,
latest-active policy, table-row evidence, and forbidden-source guards remain
authoritative.

Progress is intentionally dynamic only on an interactive terminal. Piped or
scripted commands suppress the spinner so output remains deterministic.

## JSON And Automation

```bash
# Pretty-printed JSON result objects, one per turn
printf 'Find BOM-055 Rev G\nquit\n' | uv run chat --qms-search --mode hybrid --json

# Plain scripted smoke with raw trace and full citation paths
printf 'Find BOM-055 Rev G\nquit\n' | uv run chat --qms-search --mode hybrid --plain --trace --raw-trace --full-citations
```

`--json` suppresses the startup banner, prompt text, progress, and Rich
formatting. It is the preferred mode for automation and debugging payload shape.

## Single-Turn Search

```bash
uv run search-qms "Find BOM-055 Rev G" --mode hybrid --limit 8
uv run search-qms "Find BOM-055 Rev G" --mode hybrid --limit 8 --hash-embeddings
```

`search-qms` is a single-turn command that prints the raw search result JSON. It
accepts `--mode`, `--limit`, and `--hash-embeddings`. Interactive follow-up
resolution, progress, pretty output, and prompt-prefix stripping live in
`chat --qms-search`.

## Query-Path Regression Gates

Any CLI/API/frontend change that affects query planning, query expansion,
retrieval, reranking, citation assembly, answer synthesis, artifact contracts, or
trace formatting must run the source-truth gate before broader tests:

```bash
uv run pytest evals/test_qms_source_truth_contracts.py -q
uv run pytest -q
```

Also run the targeted generated/audit eval that matches the change:

```bash
uv run search-evals --dataset smoke --mode hybrid --fail-under 0.95
uv run search-evals --dataset smoke --answers-jsonl evals/datasets/qms_smoke_golden_answers.jsonl
uv run search-evals --dataset core --validate-only
```

Use smoke/core `--report docs/eval-runs` when the run is part of a handoff. A
passing generated report does not replace the 14-query source-truth gate.

## Relevant Files

```text
src/agent/
├── cli.py          # generic and QMS chat REPL, renderer, progress, trace output
├── core.py         # generic LangChain agent factory
└── search/
    ├── cli.py      # single-turn search/index/status commands
    └── service.py  # QMS retrieval and answer orchestration
```
