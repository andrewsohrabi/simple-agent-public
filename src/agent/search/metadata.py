from __future__ import annotations

import re
from pathlib import Path

from agent.search.schema import DocumentMetadata


DOC_ID_RE = re.compile(
    r"\b("
    r"[A-Z]{2,5}-(?:P\d{2}|SWV)?-?\d{3}"
    r"|BOM-\d{3}"
    r"|ECR-\d{3}"
    r"|ESF-\d{3}"
    r"|DHF-\d{3}"
    r"|DMR-\d{3}"
    r"|DR-\d{3}"
    r"|IFU-(?:\d{3}|[A-Z0-9]+)"
    r"|QSR-\d{3}"
    r"|TRA-\d{3}"
    r"|3P-\d{3}"
    r")\b"
)
REV_SUFFIX_RE = re.compile(r"(?:^|[_\-\s])([A-Z])(?:$|[_\-\s])")
SOFTWARE_VERSION_RE = re.compile(r"\bv\d+(?:\.\d+){1,3}\b", re.IGNORECASE)
PROJECT_CODE_RE = re.compile(r"\b(P\d{2})(?:-[A-Z0-9]+)?\b", re.IGNORECASE)


def revision_rank(revision: str) -> int:
    if not revision:
        return 0
    rank = 0
    for char in revision.upper():
        if "A" <= char <= "Z":
            rank = rank * 26 + (ord(char) - ord("A") + 1)
    return rank


def document_family(doc_id: str) -> str:
    return doc_id.split("-", 1)[0].upper()


def project_code_from_filename(filename: str, doc_id: str | None = None) -> str | None:
    if doc_id:
        match = PROJECT_CODE_RE.search(doc_id)
        if match:
            return match.group(1).upper()
    match = PROJECT_CODE_RE.search(Path(filename).stem)
    return match.group(1).upper() if match else None


def _parse_revision(stem: str) -> str:
    trailing = re.search(r"_([A-Z])(?:-(?:signed|obsolete))?$", stem, re.IGNORECASE)
    if trailing:
        return trailing.group(1).upper()
    candidates = REV_SUFFIX_RE.findall(stem)
    if candidates:
        return candidates[-1].upper()
    match = re.search(r"\bRev(?:ision)?\s*([A-Z])\b", stem, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "A"


def parse_document_metadata(filename: str, source_path: str | None = None) -> DocumentMetadata:
    path = Path(filename)
    stem = path.stem
    stem_for_id = stem.replace(" ", "-")
    match = DOC_ID_RE.search(stem_for_id)
    if match:
        doc_id = match.group(1).replace("--", "-").upper()
    else:
        doc_id = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").upper()
    prefix = document_family(doc_id)
    revision = _parse_revision(stem)
    lower = stem.lower()
    is_signed = "signed" in lower
    is_obsolete = "obsolete" in lower

    title = stem
    if doc_id in title:
        title = title.replace(doc_id, "")
    title = re.sub(r"^[-_\s]+", "", title)
    title = re.sub(r"[_-]?[A-Z](?:[-_](?:signed|obsolete))?$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title.replace("_", " ")).strip(" -")
    if not title:
        title = doc_id
    version_match = SOFTWARE_VERSION_RE.search(stem)

    return DocumentMetadata(
        doc_id=doc_id,
        prefix=prefix,
        title=title,
        revision=revision,
        revision_rank=revision_rank(revision),
        canonical_doc_key=doc_id,
        is_latest=False,
        is_signed=is_signed,
        is_obsolete=is_obsolete,
        filename=path.name,
        source_path=source_path or filename,
        software_version=version_match.group(0) if version_match else None,
    )


def mark_latest(documents: list[DocumentMetadata]) -> dict[tuple[str, str], bool]:
    latest: dict[str, int] = {}
    for doc in documents:
        if doc.is_obsolete:
            latest.setdefault(doc.canonical_doc_key, doc.revision_rank)
            continue
        latest[doc.canonical_doc_key] = max(
            latest.get(doc.canonical_doc_key, 0), doc.revision_rank
        )
    return {
        (doc.doc_id, doc.revision): doc.revision_rank == latest.get(doc.canonical_doc_key)
        and not doc.is_obsolete
        for doc in documents
    }
