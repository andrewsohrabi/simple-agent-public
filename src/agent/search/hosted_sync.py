from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from agent.config import SearchConfig
from agent.search.openai_file_search import metadata_from_markdown


def sync_openai_vector_store(
    config: SearchConfig,
    normalized_dir: Path,
    *,
    corpus_hash: str = "",
    force: bool = False,
) -> dict[str, object]:
    """Create/reuse an OpenAI vector store for normalized Markdown files.

    The method intentionally stores only OpenAI file/vector-store ids and local
    source mappings. It never stores or prints API keys.
    """
    config.openai_vector_store_state.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {}
    if config.openai_vector_store_state.exists():
        state = json.loads(config.openai_vector_store_state.read_text(encoding="utf-8"))
    markdown_paths = sorted(normalized_dir.glob("*.md"))
    file_signatures = [_file_signature(path) for path in markdown_paths]
    if (
        not force
        and state.get("status") == "synced"
        and state.get("corpus_hash") == corpus_hash
        and int(state.get("file_count", 0)) == len(markdown_paths)
        and state.get("file_signatures") == file_signatures
    ):
        return state

    from openai import OpenAI

    client = OpenAI()
    vector_store = client.vector_stores.create(name="MedAI QMS normalized corpus")
    vector_store_id = vector_store.id

    files = []
    total = len(markdown_paths)
    try:
        for index, path in enumerate(markdown_paths, start=1):
            metadata = metadata_from_markdown(path)
            with path.open("rb") as handle:
                uploaded = client.files.create(file=handle, purpose="assistants")
            client.vector_stores.files.create_and_poll(
                vector_store_id=str(vector_store_id),
                file_id=uploaded.id,
                attributes={
                    "doc_id": str(metadata.get("doc_id", "")),
                    "revision": str(metadata.get("revision", "")),
                    "prefix": str(metadata.get("prefix", "")),
                    "filename": str(metadata.get("filename", path.name)),
                    "normalized_path": str(path),
                    "source_path": str(metadata.get("source_path", "")),
                },
                timeout=300,
            )
            files.append({"file_id": uploaded.id, **metadata})
            checkpoint_state = _state_payload(
                status="syncing",
                vector_store_id=str(vector_store_id),
                corpus_hash=corpus_hash,
                normalized_dir=normalized_dir,
                file_signatures=file_signatures,
                files=files,
            )
            config.openai_vector_store_state.write_text(
                json.dumps(checkpoint_state, indent=2), encoding="utf-8"
            )
            print(
                f"synced hosted file {index}/{total}: {path.name}",
                file=sys.stderr,
                flush=True,
            )
    except Exception as exc:
        failed_state = _state_payload(
            status="failed",
            vector_store_id=str(vector_store_id),
            corpus_hash=corpus_hash,
            normalized_dir=normalized_dir,
            file_signatures=file_signatures,
            files=files,
            error=f"{type(exc).__name__}: {exc}",
        )
        config.openai_vector_store_state.write_text(
            json.dumps(failed_state, indent=2), encoding="utf-8"
        )
        raise

    state = _state_payload(
        status="synced",
        vector_store_id=str(vector_store_id),
        corpus_hash=corpus_hash,
        normalized_dir=normalized_dir,
        file_signatures=file_signatures,
        files=files,
    )
    config.openai_vector_store_state.write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    return state


def _file_signature(path: Path) -> dict[str, object]:
    digest = sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "name": path.name,
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def _state_payload(
    *,
    status: str,
    vector_store_id: str,
    corpus_hash: str,
    normalized_dir: Path,
    file_signatures: list[dict[str, object]],
    files: list[dict[str, object]],
    error: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(),
        "vector_store_id": vector_store_id,
        "corpus_hash": corpus_hash,
        "normalized_dir": str(normalized_dir),
        "source_format": "normalized_markdown",
        "file_count": len(files),
        "file_signatures": file_signatures,
        "files": files,
    }
    if error:
        payload["error"] = error
    return payload
