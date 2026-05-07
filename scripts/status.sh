#!/usr/bin/env bash
set -euo pipefail

uv run search-status \
  --tasks TASKS.md \
  --index-dir .data/qms-index \
  --openai-state .data/openai/vector_store_state.json
