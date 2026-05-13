from dotenv import load_dotenv
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel, Field

from agent.config import load_config
from agent.search.service import QmsSearchService, normalize_search_mode
from agent.search.sqlite_store import SearchStore
from agent.search.stats import collect_stats

load_dotenv()
config = load_config()

app = FastAPI()
_SEARCH_SERVICE: QmsSearchService | None = None
_SEARCH_SERVICE_FACTORY = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    mode: str = "auto"
    limit: int = Field(default=16, ge=1, le=50)


class SearchRequest(BaseModel):
    query: str
    mode: str = "auto"
    limit: int = Field(default=16, ge=1, le=50)


def _search_service() -> QmsSearchService:
    global _SEARCH_SERVICE, _SEARCH_SERVICE_FACTORY
    factory = QmsSearchService
    if _SEARCH_SERVICE is None or _SEARCH_SERVICE_FACTORY is not factory:
        _SEARCH_SERVICE = factory(config, use_hash_embeddings=config.use_hash_embeddings)
        _SEARCH_SERVICE_FACTORY = factory
    return _SEARCH_SERVICE


def _store() -> SearchStore:
    return SearchStore(config.index_dir / "qms.sqlite")


def _validate_mode(mode: str) -> str:
    try:
        return normalize_search_mode(mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _chat_search_query(messages: list[Message]) -> str:
    latest_user_index = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == "user" and message.content.strip():
            latest_user_index = index
            break
    if latest_user_index is None:
        return ""

    latest_user_message = messages[latest_user_index].content.strip()
    prior_lines = [
        f"{message.role}: {message.content.strip()}"
        for message in messages[:latest_user_index]
        if message.content.strip()
    ]
    if not prior_lines:
        return latest_user_message
    return (
        "Prior conversation:\n"
        + "\n".join(prior_lines)
        + "\n\nLatest user message:\n"
        + latest_user_message
    )


def _row_to_document(row) -> dict[str, object]:
    return {
        "doc_id": row["doc_id"],
        "revision": row["revision"],
        "prefix": row["prefix"],
        "title": row["title"],
        "revision_rank": row["revision_rank"],
        "canonical_doc_key": row["canonical_doc_key"],
        "is_latest": bool(row["is_latest"]),
        "is_signed": bool(row["is_signed"]),
        "is_obsolete": bool(row["is_obsolete"]),
        "filename": row["filename"],
        "source_path": row["source_path"],
        "software_version": row["software_version"],
        "markdown_path": row["markdown_path"],
        "sha256": row["sha256"],
    }


def _row_to_chunk(row) -> dict[str, object]:
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "revision": row["revision"],
        "title": row["title"],
        "section": row["section"],
        "ordinal": row["ordinal"],
        "text": row["text"],
        "parent_section_id": row["parent_section_id"],
        "kind": row["kind"],
        "token_count": row["token_count"],
        "metadata": json.loads(row["metadata_json"]),
    }


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        search_query = _chat_search_query(req.messages)
        if not search_query:
            raise HTTPException(status_code=400, detail="user message is required")
        mode = _validate_mode(req.mode)
        service = _search_service()
        result = service.search(search_query, mode=mode, limit=req.limit)
        return {"reply": result["answer"], **result}
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    mode = _validate_mode(req.mode)
    try:
        service = _search_service()
        return service.search(req.query, mode=mode, limit=req.limit)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
def stats():
    return collect_stats(config)


@app.get("/index/status")
def index_status():
    return collect_stats(config)


@app.get("/status")
def status():
    return collect_stats(config)


@app.get("/health")
def health():
    return {"ok": True, "model_config": config.model_config()}


@app.get("/documents")
def documents(
    q: str | None = None,
    prefix: str | None = None,
    latest_only: bool = False,
    include_obsolete: bool = True,
    limit: int = Query(default=50, ge=1, le=500),
):
    clauses: list[str] = []
    values: list[object] = []
    if q:
        clauses.append("(lower(doc_id) LIKE ? OR lower(title) LIKE ? OR lower(filename) LIKE ?)")
        needle = f"%{q.lower()}%"
        values.extend([needle, needle, needle])
    if prefix:
        clauses.append("prefix = ?")
        values.append(prefix.upper())
    if latest_only:
        clauses.append("is_latest = 1")
    if not include_obsolete:
        clauses.append("is_obsolete = 0")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    try:
        with _store().connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM documents
                {where}
                ORDER BY doc_id, revision_rank DESC
                LIMIT ?
                """,
                [*values, limit],
            ).fetchall()
        return {"documents": [_row_to_document(row) for row in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/documents/{doc_id}")
def document(
    doc_id: str,
    revision: str | None = None,
    include_chunks: bool = False,
    chunk_limit: int = Query(default=25, ge=1, le=200),
):
    clauses = ["doc_id = ?"]
    values: list[object] = [doc_id.upper()]
    if revision:
        clauses.append("revision = ?")
        values.append(revision.upper())
    else:
        clauses.append("is_latest = 1")
    try:
        with _store().connect() as conn:
            row = conn.execute(
                f"""
                SELECT *
                FROM documents
                WHERE {' AND '.join(clauses)}
                ORDER BY revision_rank DESC
                LIMIT 1
                """,
                values,
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="document not found")
            payload: dict[str, object] = {"document": _row_to_document(row)}
            if include_chunks:
                chunk_rows = conn.execute(
                    """
                    SELECT *
                    FROM chunks
                    WHERE doc_id = ? AND revision = ?
                    ORDER BY ordinal
                    LIMIT ?
                    """,
                    (row["doc_id"], row["revision"], chunk_limit),
                ).fetchall()
                payload["chunks"] = [_row_to_chunk(chunk_row) for chunk_row in chunk_rows]
            return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/chunks/{chunk_id}")
def chunk(chunk_id: str):
    try:
        with _store().connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="chunk not found")
        return {"chunk": _row_to_chunk(row)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main():
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    uvicorn.run("agent.server:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
