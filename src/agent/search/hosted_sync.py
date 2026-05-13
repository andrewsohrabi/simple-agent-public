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
    state = _load_existing_state(config.openai_vector_store_state)
    markdown_paths = sorted(normalized_dir.glob("*.md"))
    file_signatures = [_file_signature(path) for path in markdown_paths]
    previous_vector_store_id = _state_vector_store_id(state)
    previous_file_ids = _state_file_ids(state)
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

    previous_cleanup = _cleanup_previous_hosted_assets(
        client,
        previous_vector_store_id=previous_vector_store_id,
        previous_file_ids=previous_file_ids,
        current_vector_store_id=str(vector_store_id),
    )
    state = _state_payload(
        status="synced",
        vector_store_id=str(vector_store_id),
        corpus_hash=corpus_hash,
        normalized_dir=normalized_dir,
        file_signatures=file_signatures,
        files=files,
        previous_cleanup=previous_cleanup,
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


def _load_existing_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def _state_vector_store_id(state: dict[str, object]) -> str | None:
    vector_store_id = state.get("vector_store_id")
    return vector_store_id if isinstance(vector_store_id, str) and vector_store_id else None


def _state_file_ids(state: dict[str, object]) -> list[str]:
    files = state.get("files")
    if not isinstance(files, list):
        return []
    file_ids: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        file_id = item.get("file_id")
        if isinstance(file_id, str) and file_id and file_id not in file_ids:
            file_ids.append(file_id)
    return file_ids


def _cleanup_previous_hosted_assets(
    client,
    *,
    previous_vector_store_id: str | None,
    previous_file_ids: list[str],
    current_vector_store_id: str,
) -> dict[str, object] | None:
    if (
        not previous_vector_store_id
        or previous_vector_store_id == current_vector_store_id
    ):
        return None

    cleanup: dict[str, object] = {
        "previous_vector_store_id": previous_vector_store_id,
        "deleted_file_ids": [],
    }
    errors: list[str] = []
    try:
        client.vector_stores.delete(previous_vector_store_id)
        cleanup["deleted_vector_store_id"] = previous_vector_store_id
    except Exception as exc:
        errors.append(f"vector_store:{type(exc).__name__}: {exc}")

    deleted_file_ids: list[str] = []
    for file_id in previous_file_ids:
        try:
            client.files.delete(file_id)
            deleted_file_ids.append(file_id)
        except Exception as exc:
            errors.append(f"file:{file_id}:{type(exc).__name__}: {exc}")
    cleanup["deleted_file_ids"] = deleted_file_ids
    if errors:
        cleanup["errors"] = errors
    return cleanup


def _state_payload(
    *,
    status: str,
    vector_store_id: str,
    corpus_hash: str,
    normalized_dir: Path,
    file_signatures: list[dict[str, object]],
    files: list[dict[str, object]],
    error: str | None = None,
    previous_cleanup: dict[str, object] | None = None,
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
    if previous_cleanup:
        payload["previous_cleanup"] = previous_cleanup
    return payload
