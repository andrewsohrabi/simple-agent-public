#!/usr/bin/env python3
"""Generate candidate QMS audit eval JSONL from a read-only SQLite index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any
from urllib.parse import quote


DEFAULT_DB = Path(".data/qms-index/qms.sqlite")
DEFAULT_LIMIT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="Path to qms.sqlite. Opened read-only with SQLite URI mode=ro.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSONL output path. Defaults to stdout and never edits datasets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum total candidate records to emit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    db_path = Path(args.db)
    records = generate_candidates(db_path, limit=args.limit)
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


def generate_candidates(db_path: Path, *, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with _connect_read_only(db_path) as conn:
        records.extend(_known_item_candidates(conn, limit=max(1, limit // 3)))
        records.extend(_count_candidates(conn, limit=max(1, limit // 3)))
        remaining = max(0, limit - len(records))
        if remaining:
            records.extend(_table_candidates(conn, limit=remaining))
    return records[:limit]


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    uri = f"file:{quote(str(resolved), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _known_item_candidates(
    conn: sqlite3.Connection,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT doc_id, revision, title
        FROM documents
        WHERE is_latest = 1 AND is_obsolete = 0
        ORDER BY prefix, doc_id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    records = []
    for index, row in enumerate(rows, start=1):
        doc_id = str(row["doc_id"])
        revision = str(row["revision"])
        title = str(row["title"])
        records.append(
            _record(
                record_id=f"qms_audit_known_item_{index:03d}",
                category="known_item_retrieval",
                prompt=(
                    f"Find {doc_id} ({title}) and return the latest active revision."
                ),
                must_include=(doc_id, f"Rev {revision}"),
                source_ids=(doc_id,),
                required_doc_ids=(doc_id,),
                citation_burden="single_source",
                revision_scope="latest",
            )
        )
    return records


def _count_candidates(
    conn: sqlite3.Connection,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT prefix, COUNT(*) AS revision_count
        FROM documents
        WHERE is_obsolete = 0
        GROUP BY prefix
        HAVING revision_count > 1
        ORDER BY revision_count DESC, prefix
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    records = []
    for index, row in enumerate(rows, start=1):
        prefix = str(row["prefix"])
        count = int(row["revision_count"])
        records.append(
            _record(
                record_id=f"qms_audit_count_{index:03d}",
                category="enumeration_counting",
                prompt=(
                    f"How many non-obsolete {prefix} document revisions are in "
                    "the SQLite inventory?"
                ),
                must_include=(f"Count: {count}", prefix),
                source_ids=(prefix,),
                required_backend="sql_inventory",
                citation_burden="metadata_inventory",
            )
        )
    return records


def _table_candidates(
    conn: sqlite3.Connection,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT doc_id, revision, title, section, text, metadata_json
        FROM chunks
        WHERE kind IN ('table_row', 'table')
          AND text LIKE '%Row %'
        ORDER BY doc_id, revision, section, ordinal
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    records = []
    for index, row in enumerate(rows, start=1):
        metadata = _loads_json_object(str(row["metadata_json"]))
        table_index = _int_or_none(metadata.get("table_index")) or _table_index(
            str(row["section"])
        )
        row_start = _int_or_none(metadata.get("row_start")) or _row_index(
            str(row["text"])
        )
        if table_index is None or row_start is None:
            continue
        doc_id = str(row["doc_id"])
        evidence = f"{doc_id} Table {table_index} row {row_start}"
        snippet = _row_snippet(str(row["text"]))
        records.append(
            _record(
                record_id=f"qms_audit_table_{index:03d}",
                category="content_extraction_synthesis",
                prompt=(
                    f"What does {doc_id} Rev {row['revision']} Table {table_index} "
                    f"row {row_start} say?"
                ),
                must_include=(evidence, snippet),
                source_ids=(doc_id,),
                required_table_evidence=(evidence,),
                citation_burden="table_or_section",
            )
        )
    return records


def _record(
    *,
    record_id: str,
    category: str,
    prompt: str,
    must_include: tuple[str, ...],
    source_ids: tuple[str, ...],
    citation_burden: str,
    revision_scope: str = "all",
    required_backend: str | None = None,
    required_doc_ids: tuple[str, ...] = (),
    required_table_evidence: tuple[str, ...] = (),
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "must_include": list(must_include),
        "source_ids": list(source_ids),
        "answer_type": "source_grounded_answer",
    }
    if required_backend:
        expected["required_backend"] = required_backend
    if required_doc_ids:
        expected["required_doc_ids"] = list(required_doc_ids)
    if required_table_evidence:
        expected["required_table_evidence"] = list(required_table_evidence)
    return {
        "id": record_id,
        "category": category,
        "prompt": prompt,
        "expected": expected,
        "tags": ["qms", "audit-candidate", "generated"],
        "difficulty": "basic",
        "dimensions": {
            "persona": "regulatory_affairs",
            "revision_scope": revision_scope,
            "answerability": "answerable",
            "noise": "none",
            "citation_burden": citation_burden,
        },
    }


def _loads_json_object(value: str) -> dict[str, Any]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _table_index(value: str) -> int | None:
    match = re.search(r"\bTable\s+(\d+)\b", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _row_index(value: str) -> int | None:
    match = re.search(r"\bRow\s+(\d+)\b", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _row_snippet(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if re.match(r"Row\s+\d+:", line, re.IGNORECASE):
            row_values: list[str] = []
            for row_line in lines[index + 1 :]:
                if re.match(r"Row\s+\d+:", row_line, re.IGNORECASE):
                    break
                row_values.append(row_line)
            remainder = " ".join(row_values).strip()
            return remainder[:120] if remainder else line
    return " ".join(lines[:2])[:120]


if __name__ == "__main__":
    raise SystemExit(main())
