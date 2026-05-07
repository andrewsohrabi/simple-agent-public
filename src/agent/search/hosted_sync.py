from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agent.config import SearchConfig


def sync_openai_vector_store(config: SearchConfig, normalized_dir: Path) -> dict[str, object]:
    """Create/reuse an OpenAI vector store for normalized Markdown files.

    The method intentionally stores only OpenAI file/vector-store ids and local
    source mappings. It never stores or prints API keys.
    """
    from openai import OpenAI

    client = OpenAI()
    config.openai_vector_store_state.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {}
    if config.openai_vector_store_state.exists():
        state = json.loads(config.openai_vector_store_state.read_text(encoding="utf-8"))

    vector_store_id = state.get("vector_store_id")
    if not vector_store_id:
        vector_store = client.vector_stores.create(name="MedAI QMS normalized corpus")
        vector_store_id = vector_store.id

    files = []
    for path in sorted(normalized_dir.glob("*.md")):
        with path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="assistants")
        client.vector_stores.files.create(
            vector_store_id=str(vector_store_id),
            file_id=uploaded.id,
        )
        files.append({"file_id": uploaded.id, "path": str(path)})

    state = {
        "updated_at": datetime.now(UTC).isoformat(),
        "vector_store_id": vector_store_id,
        "normalized_dir": str(normalized_dir),
        "files": files,
    }
    config.openai_vector_store_state.write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    return state
