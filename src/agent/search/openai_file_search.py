from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.config import SearchConfig
from agent.search.schema import SearchHit
from agent.search.sqlite_store import SearchStore


class OpenAIFileSearch:
    """Hosted File Search wrapper with local citation-source validation."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.state_path = config.openai_vector_store_state

    def is_available(self) -> bool:
        if not self.state_path.exists():
            return False
        try:
            state = self.load_state()
        except (OSError, json.JSONDecodeError):
            return False
        return (
            state.get("status") == "synced"
            and bool(state.get("vector_store_id"))
            and isinstance(state.get("files"), list)
        )

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            raise FileNotFoundError(f"hosted state not found: {self.state_path}")
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def search(
        self, query: str, store: SearchStore, *, limit: int = 16
    ) -> list[SearchHit]:
        state = self.load_state()
        vector_store_id = state.get("vector_store_id")
        if not vector_store_id:
            raise RuntimeError("hosted state is missing vector_store_id")

        from openai import OpenAI

        client = OpenAI()
        response = client.vector_stores.search(
            str(vector_store_id),
            query=query,
            max_num_results=_clamp_hosted_limit(limit),
        )
        file_map = {
            str(item.get("file_id")): item
            for item in state.get("files", [])
            if isinstance(item, dict) and item.get("file_id")
        }
        documents: list[dict[str, object]] = []
        hosted_by_doc: dict[tuple[str, str], dict[str, object]] = {}
        seen: set[tuple[str, str]] = set()
        for result_index, result in enumerate(_iter_search_results(response)):
            mapped = file_map.get(str(result.get("file_id") or ""))
            if not mapped:
                continue
            doc_id = str(mapped.get("doc_id") or "")
            revision = str(mapped.get("revision") or "")
            if not doc_id or not revision:
                continue
            key = (doc_id, revision)
            if key in seen:
                continue
            seen.add(key)
            hosted_debug = {
                "hosted_file_id": result.get("file_id"),
                "hosted_filename": result.get("filename"),
                "hosted_score": result.get("score"),
                "hosted_rank": result_index + 1,
                "hosted_content": result.get("content", []),
                "hosted_content_text": _hosted_content_text(result.get("content", [])),
                "hosted_annotations": result.get("annotations", []),
            }
            hosted_by_doc[key] = hosted_debug
            documents.append(
                {
                    "doc_id": doc_id,
                    "revision": revision,
                    "title": mapped.get("title", ""),
                    "filename": mapped.get("filename", ""),
                    "markdown_path": mapped.get("path"),
                    "is_latest": bool(mapped.get("is_latest", False)),
                    "is_obsolete": bool(mapped.get("is_obsolete", False)),
                }
            )
        hits: list[SearchHit] = []
        for document in documents:
            key = (str(document["doc_id"]), str(document["revision"]))
            hosted_debug = hosted_by_doc.get(key, {})
            hosted_text = str(hosted_debug.get("hosted_content_text", ""))
            hits.extend(
                store.chunks_for_hosted_result(
                    document,
                    hosted_text,
                    limit_per_doc=1,
                )
                or store.chunks_for_documents([document], limit_per_doc=1)
            )
        return [
            SearchHit(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                revision=hit.revision,
                title=hit.title,
                section=hit.section,
                text=hit.text,
                score=_hosted_score(
                    hosted_by_doc.get((hit.doc_id, hit.revision)), hit.score
                ),
                source="hosted_file_search",
                metadata={
                    **hit.metadata,
                    "hosted_vector_store_id": str(vector_store_id),
                    "hosted_state_path": str(self.state_path),
                    **hosted_by_doc.get((hit.doc_id, hit.revision), {}),
                },
                evidence_type=hit.evidence_type,
                support_level=hit.support_level,
                table_index=hit.table_index,
                row_start=hit.row_start,
                row_end=hit.row_end,
                heading_path=hit.heading_path,
                columns=hit.columns,
                row_cells=hit.row_cells,
            )
            for hit in hits
        ]


def _iter_search_results(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", response)
    results: list[dict[str, Any]] = []
    for item in data or []:
        if hasattr(item, "model_dump"):
            results.append(item.model_dump())
        elif isinstance(item, dict):
            results.append(item)
        else:
            results.append(
                {
                    "file_id": getattr(item, "file_id", None),
                    "filename": getattr(item, "filename", None),
                    "score": getattr(item, "score", None),
                    "content": getattr(item, "content", []),
                }
            )
    return [_normalize_result(item) for item in results]


def _normalize_result(item: dict[str, Any]) -> dict[str, Any]:
    content = _dump_value(item.get("content", []))
    annotations: list[dict[str, Any]] = []
    for annotation in _dump_value(item.get("annotations", [])) or []:
        if isinstance(annotation, dict):
            annotations.append(annotation)
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            for annotation in part.get("annotations", []) or []:
                dumped = _dump_value(annotation)
                if isinstance(dumped, dict):
                    annotations.append(dumped)
    return {
        "file_id": item.get("file_id"),
        "filename": item.get("filename"),
        "score": item.get("score"),
        "content": content if isinstance(content, list) else [],
        "annotations": annotations,
    }


def _dump_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_dump_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump_value(item) for key, item in value.items()}
    return value


def _hosted_content_text(content: object) -> str:
    if not isinstance(content, list):
        return ""
    values: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                values.append(text)
    return "\n".join(values)


def _hosted_score(hosted: dict[str, object] | None, default: float) -> float:
    if not hosted:
        return default
    score = hosted.get("hosted_score")
    if isinstance(score, int | float):
        return float(score)
    return default


def _clamp_hosted_limit(limit: int) -> int:
    return min(max(int(limit), 1), 50)


def metadata_from_markdown(path: Path) -> dict[str, object]:
    """Parse the normalized Markdown metadata preamble used for hosted state."""

    metadata: dict[str, object] = {"path": str(path)}
    if not path.exists():
        return metadata
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:48]:
        if line.startswith("# "):
            metadata["heading"] = line.removeprefix("# ").strip()
        if not line.startswith("- "):
            continue
        key, _, value = line.removeprefix("- ").partition(":")
        normalized_key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if normalized_key == "document_id":
            metadata["doc_id"] = value
        elif normalized_key == "revision":
            metadata["revision"] = value
        elif normalized_key == "prefix":
            metadata["prefix"] = value
        elif normalized_key == "latest_revision":
            metadata["is_latest"] = value.lower() == "true"
        elif normalized_key == "signed":
            metadata["is_signed"] = value.lower() == "true"
        elif normalized_key == "obsolete":
            metadata["is_obsolete"] = value.lower() == "true"
        elif normalized_key == "source_filename":
            metadata["filename"] = value
        elif normalized_key == "source_path":
            metadata["source_path"] = value
    if "title" not in metadata and "heading" in metadata:
        heading = str(metadata["heading"])
        _, _, tail = heading.partition(":")
        metadata["title"] = tail.strip() if tail else heading
    return metadata
