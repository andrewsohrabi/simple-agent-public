# Active Bugs

Use this file for unresolved or in-progress bugs during the MedAI QMS search
implementation.

When a bug is resolved and likely to recur, move or summarize the final lesson in
`docs/bugs/known_failures.md`.

## Template

```md
## <YYYY-MM-DD> - <short failure name>

- Status: open | investigating | blocked | resolved
- Owner:
- Priority:
- Command:
- Failure signature:
- Context:
- Attempts:
- Current hypothesis:
- Next action:
- Links:
```

## Open Issues

No active implementation bugs are currently unresolved.

## 2026-05-07 - `search-evals` Console Script Could Not Import `evals`

- Status: resolved
- Owner: Codex
- Priority: medium
- Command: `uv run search-evals --dataset evals/datasets/qms_core.jsonl --report docs/eval-runs --hash-embeddings --mode local`
- Failure signature: `ModuleNotFoundError: No module named 'evals'`
- Context: `pyproject.toml` registered `search-evals = "evals.run_search_evals:main"`, but Hatch only included `src/agent` in wheel packages.
- Attempts: initial eval command failed immediately before loading the dataset.
- Confirmed fix: include the root `evals` package in `[tool.hatch.build.targets.wheel].packages`.
- Verification: rerun the same `uv run search-evals ...` command after the packaging fix.
