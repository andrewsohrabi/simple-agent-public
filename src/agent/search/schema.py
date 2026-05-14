from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DocumentMetadata:
    doc_id: str
    prefix: str
    title: str
    revision: str
    revision_rank: int
    canonical_doc_key: str
    is_latest: bool
    is_signed: bool
    is_obsolete: bool
    filename: str
    source_path: str
    software_version: str | None = None


@dataclass
class NormalizedDocument:
    metadata: DocumentMetadata
    markdown: str
    markdown_path: Path | None = None
    sha256: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    revision: str
    title: str
    text: str
    search_text: str
    section: str
    ordinal: int
    metadata: dict[str, object]
    chunk_index: int = 0
    parent_section_id: str = ""
    heading_path: tuple[str, ...] = ()
    kind: str = "prose"
    ordinal_start: int = 0
    ordinal_end: int = 0
    token_count: int = 0
    evidence_type: str = "prose"
    support_level: str = "chunk"
    table_index: int | None = None
    row_start: int | None = None
    row_end: int | None = None
    columns: tuple[str, ...] = ()
    row_cells: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    revision: str
    title: str
    section: str
    text: str
    score: float
    source: str
    metadata: dict[str, object]
    evidence_type: str = "prose"
    support_level: str = "chunk"
    table_index: int | None = None
    row_start: int | None = None
    row_end: int | None = None
    heading_path: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    row_cells: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    doc_id: str
    revision: str
    title: str
    section: str
    filename: str
    markdown_path: str | None = None
    markdown_path_abs: str | None = None
    source_path: str | None = None
    source_path_abs: str | None = None
    chunk_id: str | None = None
    evidence_type: str = "metadata"
    support_level: str = "document"
    heading_path: tuple[str, ...] = ()
    table_index: int | None = None
    row_start: int | None = None
    row_end: int | None = None
    columns: tuple[str, ...] = ()
    row_cells: dict[str, str] = field(default_factory=dict)
