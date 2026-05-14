from __future__ import annotations

import json
import re
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

CREATE TABLE IF NOT EXISTS revisions (
  canonical_doc_key TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  revision TEXT NOT NULL,
  revision_rank INTEGER NOT NULL,
  is_latest INTEGER NOT NULL,
  is_obsolete INTEGER NOT NULL,
  filename TEXT NOT NULL,
  PRIMARY KEY (doc_id, revision)
);

CREATE TABLE IF NOT EXISTS doc_references (
  source_doc_id TEXT NOT NULL,
  source_revision TEXT NOT NULL,
  target_doc_id TEXT NOT NULL,
  reference_text TEXT NOT NULL,
  source TEXT NOT NULL,
  PRIMARY KEY (source_doc_id, source_revision, target_doc_id, reference_text)
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


def _remove_sqlite_file_set(path: Path) -> None:
    for suffix in ("", "-journal", "-wal", "-shm"):
        path.with_name(f"{path.name}{suffix}").unlink(missing_ok=True)


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
    REQUIRED_TABLES = {
        "documents",
        "chunks",
        "source_files",
        "ingest_runs",
        "revisions",
        "doc_references",
        "chunk_fts",
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
            table_rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')"
            ).fetchall()
            tables = {str(row["name"]) for row in table_rows}
            if not self.REQUIRED_TABLES.issubset(tables):
                return False
            rows = conn.execute("PRAGMA table_info(chunks)").fetchall()
        columns = {str(row["name"]) for row in rows}
        return self.REQUIRED_CHUNK_COLUMNS.issubset(columns)

    def load_manifest(
        self,
        manifest: dict[str, object],
        *,
        config=None,
    ) -> list[Chunk]:
        chunks = chunks_from_manifest(manifest, config=config)
        temp_path = self.db_path.with_name(f"{self.db_path.name}.tmp")
        _remove_sqlite_file_set(temp_path)
        temp_store = SearchStore(temp_path)
        try:
            temp_store.initialize()
            temp_store._replace_manifest_contents(manifest, chunks)
            temp_path.replace(self.db_path)
        except Exception:
            _remove_sqlite_file_set(temp_path)
            raise
        return chunks

    def _replace_manifest_contents(
        self,
        manifest: dict[str, object],
        chunks: list[Chunk],
    ) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunk_fts")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM source_files")
            conn.execute("DELETE FROM revisions")
            conn.execute("DELETE FROM doc_references")
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
                conn.execute(
                    """
                    INSERT OR REPLACE INTO revisions
                    (canonical_doc_key, doc_id, revision, revision_rank, is_latest, is_obsolete, filename)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["canonical_doc_key"],
                        item["doc_id"],
                        item["revision"],
                        item["revision_rank"],
                        int(bool(item["is_latest"])),
                        int(bool(item["is_obsolete"])),
                        item["filename"],
                    ),
                )
                for target_doc_id, reference_text in _extract_references(item):
                    if target_doc_id == item["doc_id"]:
                        continue
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO doc_references
                        (source_doc_id, source_revision, target_doc_id, reference_text, source)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            item["doc_id"],
                            item["revision"],
                            target_doc_id,
                            reference_text,
                            "normalized_markdown",
                        ),
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
            references = conn.execute("SELECT COUNT(*) FROM doc_references").fetchone()[0]
            return {
                "exists": True,
                "documents": docs,
                "chunks": chunks,
                "latest_documents": latest,
                "references": references,
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
            clauses.append("doc_id = ?")
            values.append(doc_id.upper())
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

    def count_documents(
        self,
        *,
        doc_id: str | None = None,
        prefix: str | None = None,
        title: str | None = None,
        revision: str | None = None,
        latest_only: bool = True,
        include_obsolete: bool = False,
    ) -> int:
        clauses: list[str] = []
        values: list[object] = []
        if doc_id:
            clauses.append("doc_id = ?")
            values.append(doc_id.upper())
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
        with self.connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM documents {where}", values).fetchone()
        return int(row[0]) if row else 0

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

    def documents_by_ids(
        self,
        doc_ids: list[str],
        *,
        latest_only: bool = True,
        include_obsolete: bool = False,
        limit_per_id: int = 10,
    ) -> list[dict[str, object]]:
        documents: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for doc_id in doc_ids:
            for doc in self.find_documents(
                doc_id=doc_id,
                latest_only=latest_only,
                include_obsolete=include_obsolete,
                limit=limit_per_id,
            ):
                key = (str(doc["doc_id"]), str(doc["revision"]))
                if key in seen:
                    continue
                documents.append(doc)
                seen.add(key)
        return documents

    def keyword_chunks(
        self,
        *,
        doc_id: str | None = None,
        prefix: str | None = None,
        all_terms: list[str] | None = None,
        any_terms: list[str] | None = None,
        latest_only: bool | None = None,
        include_obsolete: bool = False,
        limit: int = 20,
        prefer_table: bool = False,
    ) -> list[SearchHit]:
        clauses: list[str] = []
        values: list[object] = []
        if doc_id:
            clauses.append("c.doc_id = ?")
            values.append(doc_id.upper())
        if prefix:
            clauses.append("d.prefix = ?")
            values.append(prefix.upper())
        if latest_only is not None:
            clauses.append("d.is_latest = ?")
            values.append(int(latest_only))
        if not include_obsolete:
            clauses.append("d.is_obsolete = 0")
        for term in all_terms or []:
            clauses.append("lower(c.search_text) LIKE ?")
            values.append(f"%{term.lower()}%")
        if any_terms:
            any_clauses = []
            for term in any_terms:
                any_clauses.append("lower(c.search_text) LIKE ?")
                values.append(f"%{term.lower()}%")
            clauses.append("(" + " OR ".join(any_clauses) + ")")
        where = " AND ".join(clauses) if clauses else "1 = 1"
        table_order = "CASE c.kind WHEN 'table' THEN 0 ELSE 1 END," if prefer_table else ""
        values.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id AND d.revision = c.revision
                WHERE {where}
                ORDER BY {table_order} d.doc_id, d.revision_rank DESC, c.ordinal
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [_hit_from_chunk_row(row, index, source="keyword") for index, row in enumerate(rows)]

    def risk_related_documents(
        self,
        *,
        latest_only: bool = True,
        include_obsolete: bool = False,
        limit: int = 500,
    ) -> list[dict[str, object]]:
        clauses = [
            """(
                d.prefix = 'RSK'
                OR lower(d.title) LIKE '%risk%'
                OR lower(d.title) LIKE '%pfmea%'
                OR EXISTS (
                    SELECT 1
                    FROM chunks c
                    WHERE c.doc_id = d.doc_id
                      AND c.revision = d.revision
                      AND (
                        lower(c.search_text) LIKE '%risk management%'
                        OR lower(c.search_text) LIKE '%risk assessment%'
                        OR lower(c.search_text) LIKE '%risk analysis%'
                        OR lower(c.search_text) LIKE '%rmf%'
                        OR lower(c.search_text) LIKE '%pfmea%'
                      )
                )
            )"""
        ]
        values: list[object] = []
        if latest_only:
            clauses.append("d.is_latest = 1")
        if not include_obsolete:
            clauses.append("d.is_obsolete = 0")
        values.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT d.*
                FROM documents d
                WHERE {' AND '.join(clauses)}
                ORDER BY d.prefix, d.doc_id, d.revision_rank DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def risk_traced_vvpr_targets(
        self,
        *,
        source_doc_ids: tuple[str, ...] = ("VVAM-P01-004", "MEMO-P01-630"),
        limit: int = 1000,
    ) -> dict[str, object]:
        source_ids = tuple(doc_id.upper() for doc_id in source_doc_ids)
        if not source_ids:
            return {
                "indexed_doc_ids": [],
                "unindexed_doc_ids": [],
                "source_doc_ids": [],
                "trace_map": {},
            }
        placeholders = ", ".join("?" for _ in source_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.doc_id, c.revision, c.text
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id AND d.revision = c.revision
                WHERE d.is_latest = 1
                  AND d.is_obsolete = 0
                  AND c.doc_id IN ({placeholders})
                  AND lower(c.search_text) LIKE '%rsk_r%'
                  AND lower(c.search_text) LIKE '%vvpr-p01%'
                ORDER BY c.doc_id, c.revision, c.ordinal
                LIMIT ?
                """,
                (*source_ids, limit),
            ).fetchall()

        traces: dict[str, dict[str, object]] = {}
        seen_sources: set[str] = set()
        for row in rows:
            source_doc_id = str(row["doc_id"])
            seen_sources.add(source_doc_id)
            for row_text in _table_row_texts(str(row["text"])):
                risk_ids = sorted({match.group(0).upper() for match in RSK_ID_RE.finditer(row_text)})
                if not risk_ids:
                    continue
                vvpr_ids = sorted(
                    {
                        f"VVPR-P01-{match.group(1).zfill(3)}"
                        for match in VVPR_P01_RE.finditer(row_text)
                    }
                )
                if not vvpr_ids:
                    continue
                for vvpr_id in vvpr_ids:
                    trace = traces.setdefault(
                        vvpr_id,
                        {
                            "doc_id": vvpr_id,
                            "risk_ids": set(),
                            "source_doc_ids": set(),
                        },
                    )
                    trace["risk_ids"].update(risk_ids)
                    trace["source_doc_ids"].add(source_doc_id)

        if not traces:
            return {
                "indexed_doc_ids": [],
                "unindexed_doc_ids": [],
                "source_doc_ids": sorted(seen_sources),
                "trace_map": {},
            }

        target_ids = sorted(traces)
        placeholders = ", ".join("?" for _ in target_ids)
        with self.connect() as conn:
            document_rows = conn.execute(
                f"""
                SELECT doc_id
                FROM documents
                WHERE is_latest = 1
                  AND is_obsolete = 0
                  AND doc_id IN ({placeholders})
                ORDER BY doc_id
                """,
                target_ids,
            ).fetchall()
        indexed_doc_ids = sorted({str(row["doc_id"]) for row in document_rows})
        trace_map = {
            doc_id: {
                "risk_ids": sorted(trace["risk_ids"]),
                "source_doc_ids": sorted(trace["source_doc_ids"]),
            }
            for doc_id, trace in sorted(traces.items())
        }
        return {
            "indexed_doc_ids": indexed_doc_ids,
            "unindexed_doc_ids": [doc_id for doc_id in target_ids if doc_id not in indexed_doc_ids],
            "source_doc_ids": sorted(seen_sources),
            "trace_map": trace_map,
        }

    def revision_chain(
        self,
        *,
        doc_id: str | None = None,
        prefix: str | None = None,
        include_obsolete: bool = True,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        values: list[object] = []
        if doc_id:
            clauses.append("doc_id = ?")
            values.append(doc_id.upper())
        if prefix:
            clauses.append("doc_id IN (SELECT doc_id FROM documents WHERE prefix = ?)")
            values.append(prefix.upper())
        if not include_obsolete:
            clauses.append("is_obsolete = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT * FROM revisions
            {where}
            ORDER BY canonical_doc_key, revision_rank
            LIMIT ?
        """
        values.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [dict(row) for row in rows]

    def references_from(self, doc_id: str, revision: str) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM doc_references
                WHERE source_doc_id = ? AND source_revision = ?
                ORDER BY target_doc_id
                """,
                (doc_id.upper(), revision.upper()),
            ).fetchall()
        return [dict(row) for row in rows]

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
                    hits.append(_hit_from_chunk_row(row, doc_index, source="metadata"))
        return hits

    def chunks_for_hosted_result(
        self,
        document: dict[str, object],
        hosted_text: str,
        *,
        limit_per_doc: int = 1,
    ) -> list[SearchHit]:
        if not hosted_text.strip():
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE doc_id = ? AND revision = ?
                ORDER BY ordinal
                """,
                (document["doc_id"], document["revision"]),
            ).fetchall()
        ranked = _rank_hosted_rows(rows, hosted_text)
        return [
            _hit_from_chunk_row(row, index, source="metadata")
            for index, row in enumerate(ranked[:limit_per_doc])
        ]

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
                try:
                    rows = conn.execute(
                        """
                        SELECT * FROM chunks
                        WHERE lower(search_text) LIKE ?
                        LIMIT ?
                        """,
                        (f"%{query.lower().replace('-', ' ')}%", limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
        hits: list[SearchHit] = []
        for index, row in enumerate(rows):
            hits.append(_hit_from_chunk_row(row, index, source="fts"))
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
                rows = _rows_within_token_cap(
                    rows,
                    center_ordinal=chunk_index,
                    max_tokens=parent_section_max_tokens,
                )
                for row in rows:
                    if row["chunk_id"] in seen:
                        continue
                    seen.add(row["chunk_id"])
                    expanded.append(_hit_from_chunk_row(row, 0, source=hit.source, score=hit.score))
        return expanded

    def citation_for_chunk(self, chunk_id: str) -> Citation | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT c.chunk_id, c.section, c.kind, c.metadata_json,
                       d.doc_id, d.revision, d.title, d.filename,
                       d.markdown_path, d.source_path
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id AND d.revision = c.revision
                WHERE c.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        metadata = json.loads(row["metadata_json"])
        return Citation(
            doc_id=row["doc_id"],
            revision=row["revision"],
            title=row["title"],
            section=row["section"],
            filename=row["filename"],
            markdown_path=row["markdown_path"],
            markdown_path_abs=_absolute_path(row["markdown_path"]),
            source_path=row["source_path"],
            source_path_abs=_absolute_path(row["source_path"]),
            chunk_id=row["chunk_id"],
            evidence_type=_evidence_type(metadata, row["kind"]),
            support_level=str(metadata.get("support_level") or _support_level(row["kind"])),
            heading_path=tuple(str(item) for item in metadata.get("heading_path", []) if item),
            table_index=_optional_int(metadata.get("table_index")),
            row_start=_optional_int(metadata.get("row_start")),
            row_end=_optional_int(metadata.get("row_end")),
            columns=tuple(str(item) for item in metadata.get("columns", []) if item),
            row_cells={
                str(key): str(value)
                for key, value in (metadata.get("row_cells") or {}).items()
            },
        )


REFERENCE_RE = re.compile(
    r"\b(?:[A-Z0-9]{2,5}-(?:P\d{2}|SWV)?-?\d{2,3}|BOM-\d{3}|ECR-\d{3}|ESF-\d{3}|"
    r"DHF-\d{3}|DMR-\d{3}|DR-\d{3}|IFU-(?:\d{3}|[A-Z0-9]+)|QSR-\d{3}|"
    r"TRA-\d{3}|3P-(?:P\d{2}-)?\d{2,3})\b",
    re.IGNORECASE,
)
RSK_ID_RE = re.compile(r"RSK_R\d+", re.IGNORECASE)
VVPR_P01_RE = re.compile(r"VVPR-P01-\s*(\d{2,3})", re.IGNORECASE)
ROW_LABEL_RE = re.compile(r"(?m)^Row\s+\d+:\s*\n", re.IGNORECASE)
HOSTED_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,}", re.IGNORECASE)


def _rank_hosted_rows(
    rows: list[sqlite3.Row], hosted_text: str
) -> list[sqlite3.Row]:
    if not rows:
        return []
    tokens = {
        token.lower()
        for token in HOSTED_TOKEN_RE.findall(hosted_text)
        if len(token) > 2
    }
    hosted_lower = " ".join(hosted_text.lower().split())

    def score(row: sqlite3.Row) -> tuple[int, int, int]:
        text = " ".join(str(row["text"]).lower().split())
        overlap = sum(1 for token in tokens if token in text)
        phrase = int(bool(hosted_lower and hosted_lower in text))
        metadata_penalty = int(str(row["kind"]) == "metadata")
        return (phrase, overlap, -metadata_penalty)

    return sorted(rows, key=lambda row: (*score(row), -int(row["ordinal"])), reverse=True)


def _rows_within_token_cap(
    rows: list[sqlite3.Row], *, center_ordinal: int, max_tokens: int
) -> list[sqlite3.Row]:
    if not rows:
        return []
    ordered_by_distance = sorted(
        rows,
        key=lambda row: (abs(int(row["ordinal"]) - center_ordinal), int(row["ordinal"])),
    )
    if max_tokens <= 0:
        return sorted(ordered_by_distance[:1], key=lambda row: int(row["ordinal"]))
    selected: list[sqlite3.Row] = []
    total_tokens = 0
    for row in ordered_by_distance:
        row_tokens = max(int(row["token_count"]), 0)
        if selected and total_tokens + row_tokens > max_tokens:
            continue
        selected.append(row)
        total_tokens += row_tokens
    if not selected:
        selected = ordered_by_distance[:1]
    return sorted(selected, key=lambda row: int(row["ordinal"]))


def _table_row_texts(text: str) -> list[str]:
    matches = list(ROW_LABEL_RE.finditer(text))
    if not matches:
        return [text]
    rows: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        row_text = text[match.end() : end].strip()
        if row_text:
            rows.append(row_text)
    return rows


def _extract_references(item: dict[str, object]) -> list[tuple[str, str]]:
    text_parts = [
        str(item.get("filename", "")),
        str(item.get("title", "")),
        str(item.get("source_path", "")),
    ]
    markdown_path = Path(str(item.get("markdown_path", "")))
    if markdown_path.exists():
        text_parts.append(markdown_path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(text_parts)
    references: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in REFERENCE_RE.finditer(text):
        target = match.group(0).upper()
        if target in seen:
            continue
        seen.add(target)
        start = max(match.start() - 80, 0)
        end = min(match.end() + 80, len(text))
        references.append((target, " ".join(text[start:end].split())))
    return references


def _hit_from_chunk_row(
    row: sqlite3.Row,
    index: int,
    *,
    source: str,
    score: float | None = None,
) -> SearchHit:
    metadata = json.loads(row["metadata_json"])
    return SearchHit(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        revision=row["revision"],
        title=row["title"],
        section=row["section"],
        text=row["text"],
        score=score if score is not None else 1.0 / (index + 1),
        source=source,
        metadata=metadata,
        evidence_type=_evidence_type(metadata, row["kind"]),
        support_level=str(metadata.get("support_level") or _support_level(row["kind"])),
        table_index=_optional_int(metadata.get("table_index")),
        row_start=_optional_int(metadata.get("row_start")),
        row_end=_optional_int(metadata.get("row_end")),
        heading_path=tuple(str(item) for item in metadata.get("heading_path", []) if item),
        columns=tuple(str(item) for item in metadata.get("columns", []) if item),
        row_cells={
            str(key): str(value)
            for key, value in (metadata.get("row_cells") or {}).items()
        },
    )


def _evidence_type(metadata: dict[str, object], kind: str) -> str:
    value = metadata.get("evidence_type")
    if isinstance(value, str) and value:
        return value
    if kind == "metadata":
        return "metadata"
    if kind == "table_row":
        return "table_row"
    if kind == "table":
        return "table_full"
    return "prose"


def _support_level(kind: str) -> str:
    if kind == "metadata":
        return "document"
    if kind == "table_row":
        return "row"
    if kind == "table":
        return "table"
    return "chunk"


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _absolute_path(value: object) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve(strict=False))
