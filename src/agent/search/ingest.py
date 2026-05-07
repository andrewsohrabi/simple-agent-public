from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from agent.search.docx_extract import extract_docx
from agent.search.metadata import mark_latest, parse_document_metadata
from agent.search.schema import NormalizedDocument


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _markdown_for_doc(doc: NormalizedDocument) -> str:
    metadata = doc.metadata
    lines = [
        f"# {metadata.doc_id} Rev {metadata.revision}: {metadata.title}",
        "",
        "## Metadata",
        f"- Document ID: {metadata.doc_id}",
        f"- Revision: {metadata.revision}",
        f"- Prefix: {metadata.prefix}",
        f"- Latest revision: {metadata.is_latest}",
        f"- Signed: {metadata.is_signed}",
        f"- Obsolete: {metadata.is_obsolete}",
        f"- Software version: {metadata.software_version or 'unknown'}",
        f"- Source filename: {metadata.filename}",
        f"- Source path: {metadata.source_path}",
        f"- Extraction warnings: {', '.join(doc.warnings) if doc.warnings else 'none'}",
        "",
        "## Extracted Content",
        doc.markdown,
    ]
    return "\n".join(lines).strip() + "\n"


def _content_to_markdown(paragraphs: list[str], tables: list[list[list[str]]]) -> str:
    lines: list[str] = []
    if paragraphs:
        lines.extend(paragraphs)
    for index, table in enumerate(tables, start=1):
        lines.extend(["", f"### Table {index}"])
        max_width = max((len(row) for row in table), default=0)
        for row_index, row in enumerate(table):
            padded = [*row, *([""] * (max_width - len(row)))]
            lines.append("| " + " | ".join(padded) + " |")
            if row_index == 0 and max_width:
                lines.append("| " + " | ".join(["---"] * max_width) + " |")
    return "\n".join(line for line in lines if line is not None).strip()


def ingest_corpus(zip_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = output_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    for stale_markdown in normalized_dir.glob("*.md"):
        stale_markdown.unlink()

    zip_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    docs: list[NormalizedDocument] = []
    metadata_only: list[str] = []

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or "__MACOSX/" in name or Path(name).name.startswith("._"):
                continue
            if not name.lower().endswith(".docx"):
                continue
            data = archive.read(info)
            extracted = extract_docx(data)
            content = _content_to_markdown(extracted.paragraphs, extracted.tables)
            if not content.strip():
                metadata_only.append(name)
                extracted.warnings.append("empty_body_metadata_only")
            metadata = parse_document_metadata(Path(name).name, source_path=name)
            docs.append(
                NormalizedDocument(
                    metadata=metadata,
                    markdown=content,
                    sha256=hashlib.sha256(data).hexdigest(),
                    warnings=extracted.warnings,
                )
            )

    latest_flags = mark_latest([doc.metadata for doc in docs])
    normalized_docs: list[NormalizedDocument] = []
    for doc in docs:
        metadata = replace(
            doc.metadata,
            is_latest=latest_flags.get((doc.metadata.doc_id, doc.metadata.revision), False),
        )
        normalized = NormalizedDocument(
            metadata=metadata,
            markdown=doc.markdown,
            sha256=doc.sha256,
            warnings=doc.warnings,
        )
        filename = f"{_safe_name(metadata.doc_id)}_rev-{metadata.revision}.md"
        path = normalized_dir / filename
        normalized.markdown_path = path
        path.write_text(_markdown_for_doc(normalized), encoding="utf-8")
        normalized_docs.append(normalized)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_zip": str(zip_path),
        "source_sha256": zip_hash,
        "document_count": len(normalized_docs),
        "skipped_empty_count": 0,
        "metadata_only_count": len(metadata_only),
        "metadata_only": metadata_only,
        "normalized_dir": str(normalized_dir),
        "documents": [
            {
                **doc.metadata.__dict__,
                "sha256": doc.sha256,
                "markdown_path": str(doc.markdown_path),
                "warnings": doc.warnings,
            }
            for doc in normalized_docs
        ],
    }
    (output_dir / "ingest_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_ingest_manifest(index_dir: Path) -> dict[str, object] | None:
    path = index_dir / "ingest_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
