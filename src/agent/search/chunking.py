from __future__ import annotations

import hashlib
import re
from pathlib import Path

from agent.config import SearchConfig
from agent.search.schema import Chunk, DocumentMetadata


TOKEN_RE = re.compile(r"\w+(?:[-.]\w+)*|[^\w\s]", re.UNICODE)
TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|\s*$")


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower() or "section"


def _status(metadata: DocumentMetadata) -> str:
    return "Obsolete" if metadata.is_obsolete else "Active"


def metadata_preamble(metadata: DocumentMetadata, section: str) -> str:
    return (
        f"Document: {metadata.doc_id}\n"
        f"Filename: {metadata.filename}\n"
        f"Title: {metadata.title}\n"
        f"Revision: {metadata.revision}\n"
        f"Status: {_status(metadata)}\n"
        f"Signed: {str(metadata.is_signed).lower()}\n"
        f"Software version: {metadata.software_version or 'unknown'}\n"
        f"Section: {section}\n"
        "Content:\n"
    )


def metadata_chunk_text(metadata: DocumentMetadata, warnings: list[str] | None = None) -> str:
    return "\n".join(
        [
            f"Filename: {metadata.filename}",
            f"Document code: {metadata.doc_id}",
            f"Document family: {metadata.prefix}",
            f"Title: {metadata.title}",
            f"Revision: {metadata.revision}",
            f"Revision rank: {metadata.revision_rank}",
            f"Signed status: {metadata.is_signed}",
            f"Obsolete status: {metadata.is_obsolete}",
            f"Software version: {metadata.software_version or 'unknown'}",
            f"Extracted title: {metadata.title}",
            f"Extraction warnings: {', '.join(warnings or []) if warnings else 'none'}",
        ]
    )


def split_markdown_sections(markdown: str) -> list[tuple[tuple[str, ...], str]]:
    sections: list[tuple[tuple[str, ...], list[str]]] = []
    heading_stack: list[str] = ["Document"]
    current_lines: list[str] = []

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((tuple(heading_stack), current_lines.copy()))

    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            heading_stack[:] = heading_stack[: max(level - 1, 0)]
            heading_stack.append(heading)
            current_lines.clear()
        else:
            current_lines.append(line)
    flush()
    return [(path, "\n".join(lines).strip()) for path, lines in sections]


def chunk_text(
    metadata: DocumentMetadata,
    markdown: str,
    *,
    config: SearchConfig | None = None,
    warnings: list[str] | None = None,
) -> list[Chunk]:
    config = config or SearchConfig()
    chunks: list[Chunk] = []
    chunk_index = 0

    if config.create_metadata_chunks:
        raw = metadata_chunk_text(metadata, warnings)
        chunks.append(
            _make_chunk(
                metadata,
                raw_text=raw,
                section="Metadata",
                heading_path=("Metadata",),
                kind="metadata",
                chunk_index=chunk_index,
                ordinal_start=0,
                ordinal_end=0,
            )
        )
        chunk_index += 1

    sections = _merge_short_sections(
        split_markdown_sections(markdown), config.min_chunk_tokens
    )
    for heading_path, section_text in sections:
        blocks = _split_section_blocks(section_text)
        prose_buffer: list[str] = []
        prose_start = 0
        for block_index, block in enumerate(blocks):
            if _is_table_block(block):
                if prose_buffer:
                    chunk_index = _emit_prose_chunks(
                        chunks,
                        metadata,
                        "\n\n".join(prose_buffer),
                        heading_path,
                        chunk_index,
                        prose_start,
                        config,
                    )
                    prose_buffer = []
                chunk_index = _emit_table_chunks(
                    chunks,
                    metadata,
                    block,
                    heading_path,
                    chunk_index,
                    block_index,
                    config,
                )
                prose_start = block_index + 1
            else:
                if not prose_buffer:
                    prose_start = block_index
                prose_buffer.append(block)
        if prose_buffer:
            chunk_index = _emit_prose_chunks(
                chunks,
                metadata,
                "\n\n".join(prose_buffer),
                heading_path,
                chunk_index,
                prose_start,
                config,
            )
    return chunks


def _split_section_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if TABLE_ROW_RE.match(line):
            table_lines = []
            while index < len(lines) and TABLE_ROW_RE.match(lines[index]):
                table_lines.append(lines[index])
                index += 1
            blocks.append("\n".join(table_lines))
            continue
        paragraph = []
        while index < len(lines) and not TABLE_ROW_RE.match(lines[index]):
            if not lines[index].strip() and paragraph:
                index += 1
                break
            paragraph.append(lines[index])
            index += 1
        text_block = "\n".join(paragraph).strip()
        if text_block:
            blocks.append(text_block)
    return blocks


def _merge_short_sections(
    sections: list[tuple[tuple[str, ...], str]], min_tokens: int
) -> list[tuple[tuple[str, ...], str]]:
    merged: list[tuple[tuple[str, ...], str]] = []
    carry: list[tuple[tuple[str, ...], str]] = []
    for heading_path, text in sections:
        token_count = count_tokens(text)
        has_table = any(_is_table_block(block) for block in _split_section_blocks(text))
        if token_count < min_tokens and not has_table:
            carry.append((heading_path, text))
            continue
        if carry:
            carried_text = [
                f"{' > '.join(path)}\n{carried}".strip()
                for path, carried in carry
            ]
            text = "\n\n".join([*carried_text, text]).strip()
            carry = []
        merged.append((heading_path, text))
    if carry:
        if merged:
            heading_path, text = merged[-1]
            carried_text = [
                f"{' > '.join(path)}\n{carried}".strip()
                for path, carried in carry
            ]
            merged[-1] = (heading_path, f"{text}\n\n" + "\n\n".join(carried_text))
        else:
            merged.extend(carry)
    return merged


def _is_table_block(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return len(lines) >= 2 and all(TABLE_ROW_RE.match(line) for line in lines[:2])


def _emit_prose_chunks(
    chunks: list[Chunk],
    metadata: DocumentMetadata,
    text: str,
    heading_path: tuple[str, ...],
    chunk_index: int,
    ordinal_start: int,
    config: SearchConfig,
) -> int:
    units = _paragraph_units(text)
    current: list[str] = []
    current_tokens = 0
    ordinal = ordinal_start
    for unit in units:
        unit_tokens = count_tokens(unit)
        if (
            current
            and current_tokens + unit_tokens > config.chunk_size_tokens
            and current_tokens >= config.min_chunk_tokens
        ):
            raw = "\n\n".join(current).strip()
            chunk_index = _append_chunk(
                chunks,
                metadata,
                raw,
                heading_path,
                "prose",
                chunk_index,
                ordinal,
                ordinal + len(current) - 1,
            )
            overlap = _overlap_units(current, config.chunk_overlap_tokens)
            current = overlap
            current_tokens = count_tokens("\n\n".join(current))
            ordinal = max(ordinal, ordinal + len(current) - 1)
        if unit_tokens > config.max_chunk_tokens:
            split_units = _split_long_unit(unit, config.max_chunk_tokens)
            for split in split_units:
                if current and current_tokens + count_tokens(split) > config.max_chunk_tokens:
                    raw = "\n\n".join(current).strip()
                    chunk_index = _append_chunk(
                        chunks,
                        metadata,
                        raw,
                        heading_path,
                        "prose",
                        chunk_index,
                        ordinal,
                        ordinal + len(current) - 1,
                    )
                    current = []
                    current_tokens = 0
                current.append(split)
                current_tokens += count_tokens(split)
        else:
            current.append(unit)
            current_tokens += unit_tokens
    if current:
        raw = "\n\n".join(current).strip()
        if chunks and count_tokens(raw) < config.min_chunk_tokens:
            previous = chunks[-1]
            if previous.kind == "prose" and previous.parent_section_id == _parent_id(metadata, heading_path):
                merged_raw = f"{previous.text}\n\n{raw}".strip()
                chunks[-1] = _make_chunk(
                    metadata,
                    raw_text=merged_raw,
                    section=" > ".join(heading_path),
                    heading_path=heading_path,
                    kind="prose",
                    chunk_index=previous.chunk_index,
                    ordinal_start=previous.ordinal_start,
                    ordinal_end=ordinal + len(current) - 1,
                )
                return chunk_index
        chunk_index = _append_chunk(
            chunks,
            metadata,
            raw,
            heading_path,
            "prose",
            chunk_index,
            ordinal,
            ordinal + len(current) - 1,
        )
    return chunk_index


def _paragraph_units(text: str) -> list[str]:
    return [unit.strip() for unit in re.split(r"\n\s*\n", text) if unit.strip()]


def _split_long_unit(text: str, max_tokens: int) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    parts: list[str] = []
    for start in range(0, len(tokens), max_tokens):
        parts.append(" ".join(tokens[start : start + max_tokens]))
    return parts


def _overlap_units(units: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    selected: list[str] = []
    total = 0
    for unit in reversed(units):
        selected.insert(0, unit)
        total += count_tokens(unit)
        if total >= overlap_tokens:
            break
    return selected


def _emit_table_chunks(
    chunks: list[Chunk],
    metadata: DocumentMetadata,
    table: str,
    heading_path: tuple[str, ...],
    chunk_index: int,
    ordinal_start: int,
    config: SearchConfig,
) -> int:
    lines = [line for line in table.splitlines() if line.strip()]
    parsed = _parse_table_lines(lines)
    header = " | ".join(parsed["columns"])
    rows = [" | ".join(row) for row in parsed["rows"]]
    table_name = heading_path[-1] if heading_path else "Table"
    table_index = _table_index(table_name)

    if len(rows) <= 3 and count_tokens(table) <= config.table_chunk_max_tokens:
        raw = _table_text(table_name, header, rows)
        chunk_index = _append_chunk(
            chunks,
            metadata,
            raw,
            heading_path,
            "table",
            chunk_index,
            ordinal_start,
            ordinal_start + max(len(rows) - 1, 0),
            evidence_type="table_full",
            support_level="table",
            table_index=table_index,
            row_start=1 if rows else None,
            row_end=len(rows) if rows else None,
            columns=tuple(parsed["columns"]),
        )
    if len(parsed["rows"]) <= config.table_row_chunk_max_rows:
        for row_number, row_cells in enumerate(parsed["rows"], start=1):
            row_map = _row_cell_map(parsed["columns"], row_cells)
            raw = _table_row_text(table_name, row_number, row_map)
            chunk_index = _append_chunk(
                chunks,
                metadata,
                raw,
                heading_path,
                "table_row",
                chunk_index,
                ordinal_start + row_number - 1,
                ordinal_start + row_number - 1,
                evidence_type="table_row",
                support_level="row",
                table_index=table_index,
                row_start=row_number,
                row_end=row_number,
                columns=tuple(parsed["columns"]),
                row_cells=row_map,
            )

    if len(rows) <= 3:
        return chunk_index

    if len(rows) > 3 and count_tokens(table) <= config.table_chunk_max_tokens:
        raw = _table_text(table_name, header, rows)
        return _append_chunk(
            chunks,
            metadata,
            raw,
            heading_path,
            "table",
            chunk_index,
            ordinal_start,
            ordinal_start + max(len(rows) - 1, 0),
            evidence_type="table_full",
            support_level="table",
            table_index=table_index,
            row_start=1 if rows else None,
            row_end=len(rows) if rows else None,
            columns=tuple(parsed["columns"]),
        )

    current_rows: list[str] = []
    current_tokens = count_tokens(header)
    row_number = 1
    for row in rows:
        row_tokens = count_tokens(row)
        if current_rows and current_tokens + row_tokens > config.table_chunk_target_tokens:
            raw = _table_text(table_name, header, current_rows, start_row=row_number - len(current_rows))
            chunk_index = _append_chunk(
                chunks,
                metadata,
                raw,
                heading_path,
                "table",
                chunk_index,
                ordinal_start + row_number - len(current_rows),
                ordinal_start + row_number - 1,
                evidence_type="table_full",
                support_level="row_group",
                table_index=table_index,
                row_start=row_number - len(current_rows),
                row_end=row_number - 1,
                columns=tuple(parsed["columns"]),
            )
            current_rows = []
            current_tokens = count_tokens(header)
        current_rows.append(row)
        current_tokens += row_tokens
        row_number += 1
    if current_rows:
        raw = _table_text(table_name, header, current_rows, start_row=row_number - len(current_rows))
        chunk_index = _append_chunk(
            chunks,
            metadata,
            raw,
            heading_path,
            "table",
            chunk_index,
            ordinal_start + row_number - len(current_rows),
            ordinal_start + row_number - 1,
            evidence_type="table_full",
            support_level="row_group",
            table_index=table_index,
            row_start=row_number - len(current_rows),
            row_end=row_number - 1,
            columns=tuple(parsed["columns"]),
        )
    return chunk_index


def _parse_table_lines(lines: list[str]) -> dict[str, list[list[str]] | list[str]]:
    header_index = _markdown_header_row_index(lines)
    rows = [_split_table_row(line) for line in lines if not TABLE_SEPARATOR_RE.match(line)]
    if not rows:
        return {"columns": [], "rows": []}
    if header_index is None:
        body_rows = rows
        columns = [
            f"Column {index + 1}"
            for index in range(max((len(row) for row in body_rows), default=0))
        ]
    else:
        columns = [
            _clean_column(cell, index) for index, cell in enumerate(rows[header_index])
        ]
        body_rows = rows[header_index + 1 :]
        if not columns:
            columns = [
                f"Column {index + 1}"
                for index in range(max((len(row) for row in body_rows), default=0))
            ]
    normalized_rows = [
        [*row, *([""] * (len(columns) - len(row)))]
        for row in body_rows
    ]
    return {"columns": columns, "rows": [row[: len(columns)] for row in normalized_rows]}


def _markdown_header_row_index(lines: list[str]) -> int | None:
    non_separator_rows = 0
    for line in lines:
        if TABLE_SEPARATOR_RE.match(line):
            return max(non_separator_rows - 1, 0)
        non_separator_rows += 1
    return None


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _clean_column(cell: str, index: int) -> str:
    value = " ".join(cell.split())
    return value or f"Column {index + 1}"


def _row_cell_map(columns: list[str], row: list[str]) -> dict[str, str]:
    cells = [*row, *([""] * (len(columns) - len(row)))]
    return {column: cells[index].strip() for index, column in enumerate(columns)}


def _table_row_text(table_name: str, row_number: int, row_cells: dict[str, str]) -> str:
    values = " | ".join(f"{column}={value}" for column, value in row_cells.items())
    return f"Table: {table_name}\nRow {row_number}: {values}"


def _table_index(table_name: str) -> int | None:
    match = re.search(r"\bTable\s+(\d+)\b", table_name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _format_table_row(line: str) -> str:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return " | ".join(cells)


def _table_text(
    table_name: str, header: str, rows: list[str], *, start_row: int = 1
) -> str:
    lines = [f"Table: {table_name}", f"Columns: {header}"]
    for offset, row in enumerate(rows, start=start_row):
        lines.extend([f"Row {offset}:", row])
    return "\n".join(lines)


def _append_chunk(
    chunks: list[Chunk],
    metadata: DocumentMetadata,
    raw: str,
    heading_path: tuple[str, ...],
    kind: str,
    chunk_index: int,
    ordinal_start: int,
    ordinal_end: int,
    evidence_type: str | None = None,
    support_level: str | None = None,
    table_index: int | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
    columns: tuple[str, ...] = (),
    row_cells: dict[str, str] | None = None,
) -> int:
    chunks.append(
        _make_chunk(
            metadata,
            raw_text=raw,
            section=" > ".join(heading_path),
            heading_path=heading_path,
            kind=kind,
            chunk_index=chunk_index,
            ordinal_start=ordinal_start,
            ordinal_end=ordinal_end,
            evidence_type=evidence_type,
            support_level=support_level,
            table_index=table_index,
            row_start=row_start,
            row_end=row_end,
            columns=columns,
            row_cells=row_cells,
        )
    )
    return chunk_index + 1


def _make_chunk(
    metadata: DocumentMetadata,
    *,
    raw_text: str,
    section: str,
    heading_path: tuple[str, ...],
    kind: str,
    chunk_index: int,
    ordinal_start: int,
    ordinal_end: int,
    evidence_type: str | None = None,
    support_level: str | None = None,
    table_index: int | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
    columns: tuple[str, ...] = (),
    row_cells: dict[str, str] | None = None,
) -> Chunk:
    parent_section_id = _parent_id(metadata, heading_path)
    search_text = metadata_preamble(metadata, section) + raw_text
    chunk_id = hashlib.sha256(
        f"{metadata.doc_id}:{metadata.revision}:{kind}:{chunk_index}:{section}:{raw_text}".encode()
    ).hexdigest()[:20]
    token_count = count_tokens(raw_text)
    evidence_type = evidence_type or _evidence_type_for_kind(kind)
    support_level = support_level or _support_level_for_kind(kind)
    row_cells = row_cells or {}
    base_metadata = {
        "prefix": metadata.prefix,
        "is_latest": metadata.is_latest,
        "is_signed": metadata.is_signed,
        "is_obsolete": metadata.is_obsolete,
        "filename": metadata.filename,
        "source_path": metadata.source_path,
        "document_code": metadata.doc_id,
        "software_version": metadata.software_version,
        "parent_section_id": parent_section_id,
        "heading_path": list(heading_path),
        "kind": kind,
        "chunk_index": chunk_index,
        "ordinal_start": ordinal_start,
        "ordinal_end": ordinal_end,
        "token_count": token_count,
        "evidence_type": evidence_type,
        "support_level": support_level,
        "table_index": table_index,
        "row_start": row_start,
        "row_end": row_end,
        "columns": list(columns),
        "row_cells": row_cells,
    }
    return Chunk(
        chunk_id=chunk_id,
        doc_id=metadata.doc_id,
        revision=metadata.revision,
        title=metadata.title,
        text=raw_text,
        search_text=search_text,
        section=section,
        ordinal=chunk_index,
        metadata=base_metadata,
        chunk_index=chunk_index,
        parent_section_id=parent_section_id,
        heading_path=heading_path,
        kind=kind,
        ordinal_start=ordinal_start,
        ordinal_end=ordinal_end,
        token_count=token_count,
        evidence_type=evidence_type,
        support_level=support_level,
        table_index=table_index,
        row_start=row_start,
        row_end=row_end,
        columns=columns,
        row_cells=row_cells,
    )


def _evidence_type_for_kind(kind: str) -> str:
    if kind == "metadata":
        return "metadata"
    if kind == "table":
        return "table_full"
    if kind == "table_row":
        return "table_row"
    return "prose"


def _support_level_for_kind(kind: str) -> str:
    if kind == "metadata":
        return "document"
    if kind == "table":
        return "table"
    if kind == "table_row":
        return "row"
    return "chunk"


def _parent_id(metadata: DocumentMetadata, heading_path: tuple[str, ...]) -> str:
    return hashlib.sha256(
        f"{metadata.doc_id}:{metadata.revision}:{' > '.join(heading_path)}".encode()
    ).hexdigest()[:16]


def chunks_from_manifest(manifest: dict[str, object], *, config: SearchConfig | None = None) -> list[Chunk]:
    config = config or SearchConfig()
    chunks: list[Chunk] = []
    for item in manifest.get("documents", []):
        if not isinstance(item, dict):
            continue
        path_value = item.get("markdown_path")
        if not path_value:
            continue
        markdown_path = Path(str(path_value))
        if not markdown_path.exists():
            continue
        metadata = DocumentMetadata(
            doc_id=str(item["doc_id"]),
            prefix=str(item["prefix"]),
            title=str(item["title"]),
            revision=str(item["revision"]),
            revision_rank=int(item["revision_rank"]),
            canonical_doc_key=str(item["canonical_doc_key"]),
            is_latest=bool(item["is_latest"]),
            is_signed=bool(item["is_signed"]),
            is_obsolete=bool(item["is_obsolete"]),
            filename=str(item["filename"]),
            source_path=str(item["source_path"]),
            software_version=item.get("software_version"),
        )
        chunks.extend(
            chunk_text(
                metadata,
                markdown_path.read_text(encoding="utf-8"),
                config=config,
                warnings=list(item.get("warnings", [])),
            )
        )
    return chunks
