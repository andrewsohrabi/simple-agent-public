from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent.search.chunking import chunks_from_manifest
from agent.search.schema import Citation, Chunk, SearchHit


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT NOT NULL,
  revision TEXT NOT NULL,
  prefix TEXT NOT NULL,
  title TEXT NOT NULL,
  revision_rank INTEGER NOT NULL,
  canonical_doc_key TEXT NOT NULL,
  is_latest INTEGER NOT NULL,
  is_signed INTEGER NOT NULL,
  is_obsolete INTEGER NOT NULL,
  filename TEXT NOT NULL,
  source_path TEXT NOT NULL,
  software_version TEXT,
  markdown_path TEXT,
  sha256 TEXT,
  PRIMARY KEY (doc_id, revision)
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  revision TEXT NOT NULL,
  title TEXT NOT NULL,
  section TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  text TEXT NOT NULL,
  search_text TEXT NOT NULL,
  parent_section_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  token_count INTEGER NOT NULL,
  metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_files (
  source_path TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  sha256 TEXT
);

CREATE TABLE IF NOT EXISTS ingest_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  source_zip TEXT NOT NULL,
  source_sha256 TEXT NOT NULL,
  document_count INTEGER NOT NULL,
  skipped_empty_count INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
  chunk_id UNINDEXED,
  doc_id,
  revision,
  title,
  section,
  text
);
"""


class SearchStore:
    REQUIRED_CHUNK_COLUMNS = {
        "chunk_id",
        "doc_id",
        "revision",
        "title",
        "section",
        "ordinal",
        "text",
        "search_text",
        "parent_section_id",
        "kind",
        "token_count",
        "metadata_json",
    }

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def schema_current(self) -> bool:
        if not self.db_path.exists():
            return False
        with self.connect() as conn:
            rows = conn.execute("PRAGMA table_info(chunks)").fetchall()
        columns = {str(row["name"]) for row in rows}
        return self.REQUIRED_CHUNK_COLUMNS.issubset(columns)

    def load_manifest(
        self,
        manifest: dict[str, object],
        *,
        config=None,
    ) -> list[Chunk]:
        if self.db_path.exists():
            self.db_path.unlink()
        self.initialize()
        chunks = chunks_from_manifest(manifest, config=config)
        with self.connect() as conn:
            conn.execute("DELETE FROM chunk_fts")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM source_files")
            conn.execute(
                """
                INSERT INTO ingest_runs
                (created_at, source_zip, source_sha256, document_count, skipped_empty_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    manifest.get("created_at", ""),
                    manifest.get("source_zip", ""),
                    manifest.get("source_sha256", ""),
                    manifest.get("document_count", 0),
                    manifest.get("skipped_empty_count", 0),
                ),
            )
            for item in manifest.get("documents", []):
                if not isinstance(item, dict):
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents
                    (doc_id, revision, prefix, title, revision_rank, canonical_doc_key,
                     is_latest, is_signed, is_obsolete, filename, source_path, software_version,
                     markdown_path, sha256)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["doc_id"],
                        item["revision"],
                        item["prefix"],
                        item["title"],
                        item["revision_rank"],
                        item["canonical_doc_key"],
                        int(bool(item["is_latest"])),
                        int(bool(item["is_signed"])),
                        int(bool(item["is_obsolete"])),
                        item["filename"],
                        item["source_path"],
                        item.get("software_version"),
                        item.get("markdown_path"),
                        item.get("sha256"),
                    ),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO source_files (source_path, filename, sha256) VALUES (?, ?, ?)",
                    (item["source_path"], item["filename"], item.get("sha256")),
                )
            for chunk in chunks:
                metadata_json = json.dumps(chunk.metadata, sort_keys=True)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, revision, title, section, ordinal, text, search_text,
                     parent_section_id, kind, token_count, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.revision,
                        chunk.title,
                        chunk.section,
                        chunk.ordinal,
                        chunk.text,
                        chunk.search_text,
                        chunk.parent_section_id,
                        chunk.kind,
                        chunk.token_count,
                        metadata_json,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO chunk_fts (chunk_id, doc_id, revision, title, section, text)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.revision,
                        chunk.title,
                        chunk.section,
                        chunk.search_text,
                    ),
                )
        return chunks

    def stats(self) -> dict[str, object]:
        if not self.db_path.exists():
            return {
                "exists": False,
                "documents": 0,
                "chunks": 0,
                "schema_current": False,
                "requires_rebuild": True,
            }
        if not self.schema_current():
            return {
                "exists": True,
                "documents": 0,
                "chunks": 0,
                "schema_current": False,
                "requires_rebuild": True,
            }
        with self.connect() as conn:
            docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            latest = conn.execute("SELECT COUNT(*) FROM documents WHERE is_latest = 1").fetchone()[0]
            return {
                "exists": True,
                "documents": docs,
                "chunks": chunks,
                "latest_documents": latest,
                "schema_current": True,
                "requires_rebuild": False,
            }

    def find_documents(
        self,
        *,
        doc_id: str | None = None,
        prefix: str | None = None,
        title: str | None = None,
        revision: str | None = None,
        latest_only: bool = True,
        include_obsolete: bool = False,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        values: list[object] = []
        if doc_id:
            clauses.append("doc_id LIKE ?")
            values.append(f"%{doc_id.upper()}%")
        if prefix:
            clauses.append("prefix = ?")
            values.append(prefix.upper())
        if title:
            clauses.append("lower(title) LIKE ?")
            values.append(f"%{title.lower()}%")
        if revision:
            clauses.append("revision = ?")
            values.append(revision.upper())
        elif latest_only:
            clauses.append("is_latest = 1")
        if not include_obsolete:
            clauses.append("is_obsolete = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT * FROM documents
            {where}
            ORDER BY doc_id, revision_rank DESC
            LIMIT ?
        """
        values.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [dict(row) for row in rows]

    def count_by_prefix(self, prefix: str, include_obsolete: bool = False) -> dict[str, object]:
        clauses = ["prefix = ?"]
        values: list[object] = [prefix.upper()]
        if not include_obsolete:
            clauses.append("is_obsolete = 0")
        where = " AND ".join(clauses)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM documents WHERE {where} ORDER BY doc_id, revision_rank DESC",
                values,
            ).fetchall()
        docs = [dict(row) for row in rows]
        return {"prefix": prefix.upper(), "count": len(docs), "documents": docs}

    def chunks_for_documents(
        self, documents: list[dict[str, object]], *, limit_per_doc: int = 1
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        with self.connect() as conn:
            for doc_index, doc in enumerate(documents):
                rows = conn.execute(
                    """
                    SELECT * FROM chunks
                    WHERE doc_id = ? AND revision = ?
                    ORDER BY CASE kind WHEN 'metadata' THEN 0 ELSE 1 END, ordinal
                    LIMIT ?
                    """,
                    (doc["doc_id"], doc["revision"], limit_per_doc),
                ).fetchall()
                for row in rows:
                    hits.append(
                        SearchHit(
                            chunk_id=row["chunk_id"],
                            doc_id=row["doc_id"],
                            revision=row["revision"],
                            title=row["title"],
                            section=row["section"],
                            text=row["text"],
                            score=1.0 / (doc_index + 1),
                            source="metadata",
                            metadata=json.loads(row["metadata_json"]),
                        )
                    )
        return hits

    def fts_search(self, query: str, *, limit: int = 20) -> list[SearchHit]:
        if not query.strip():
            return []
        tokens = [
            token
            for token in query.replace("-", " ").split()
            if token.strip().replace("_", "").isalnum()
        ]
        escaped = " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)
        with self.connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT c.*, bm25(chunk_fts) AS rank
                    FROM chunk_fts
                    JOIN chunks c USING (chunk_id)
                    WHERE chunk_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (escaped or query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            if not rows:
                rows = conn.execute(
                    """
                    SELECT * FROM chunks
                    WHERE lower(search_text) LIKE ?
                    LIMIT ?
                    """,
                    (f"%{query.lower().replace('-', ' ')}%", limit),
                ).fetchall()
        hits: list[SearchHit] = []
        for index, row in enumerate(rows):
            metadata = json.loads(row["metadata_json"])
            hits.append(
                SearchHit(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    revision=row["revision"],
                    title=row["title"],
                    section=row["section"],
                    text=row["text"],
                    score=1.0 / (index + 1),
                    source="fts",
                    metadata=metadata,
                )
            )
        return hits

    def expand_neighbors(
        self,
        hits: list[SearchHit],
        *,
        neighbor_chunks: int,
        parent_section_max_tokens: int,
    ) -> list[SearchHit]:
        if neighbor_chunks <= 0:
            return hits
        expanded: list[SearchHit] = []
        seen: set[str] = set()
        with self.connect() as conn:
            for hit in hits:
                metadata = hit.metadata
                parent_id = str(metadata.get("parent_section_id", ""))
                chunk_index = int(metadata.get("chunk_index", 0))
                lower = chunk_index - neighbor_chunks
                upper = chunk_index + neighbor_chunks
                rows = conn.execute(
                    """
                    SELECT * FROM chunks
                    WHERE doc_id = ? AND revision = ? AND parent_section_id = ?
                      AND ordinal BETWEEN ? AND ?
                    ORDER BY ordinal
                    """,
                    (hit.doc_id, hit.revision, parent_id, lower, upper),
                ).fetchall()
                total_tokens = sum(int(row["token_count"]) for row in rows)
                if total_tokens > parent_section_max_tokens:
                    rows = [
                        row for row in rows if abs(int(row["ordinal"]) - chunk_index) <= neighbor_chunks
                    ]
                for row in rows:
                    if row["chunk_id"] in seen:
                        continue
                    seen.add(row["chunk_id"])
                    expanded.append(
                        SearchHit(
                            chunk_id=row["chunk_id"],
                            doc_id=row["doc_id"],
                            revision=row["revision"],
                            title=row["title"],
                            section=row["section"],
                            text=row["text"],
                            score=hit.score,
                            source=hit.source,
                            metadata=json.loads(row["metadata_json"]),
                        )
                    )
        return expanded

    def citation_for_chunk(self, chunk_id: str) -> Citation | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT c.chunk_id, c.section, d.doc_id, d.revision, d.title, d.filename, d.markdown_path
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id AND d.revision = c.revision
                WHERE c.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return Citation(
            doc_id=row["doc_id"],
            revision=row["revision"],
            title=row["title"],
            section=row["section"],
            filename=row["filename"],
            markdown_path=row["markdown_path"],
            chunk_id=row["chunk_id"],
        )
