from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET


WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass
class ExtractedDocx:
    paragraphs: list[str] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def plain_text(self) -> str:
        table_lines = []
        for table in self.tables:
            for row in table:
                table_lines.append(" | ".join(cell for cell in row if cell))
        return "\n".join([*self.paragraphs, *table_lines]).strip()


def _text_from_element(element: ET.Element) -> str:
    values = [node.text or "" for node in element.iter(f"{WORD_NS}t")]
    return "".join(values).strip()


def _parse_table(table_el: ET.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_el in table_el.findall(f"{WORD_NS}tr"):
        row: list[str] = []
        for cell_el in row_el.findall(f"{WORD_NS}tc"):
            cell = " ".join(
                text
                for text in (_text_from_element(p) for p in cell_el.findall(f"{WORD_NS}p"))
                if text
            )
            row.append(cell.strip())
        if any(row):
            rows.append(row)
    return rows


def extract_docx(data: bytes) -> ExtractedDocx:
    extracted = ExtractedDocx()
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            document_xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        extracted.warnings.append(f"Could not read DOCX document.xml: {exc}")
        return extracted

    root = ET.fromstring(document_xml)
    body = root.find(f"{WORD_NS}body")
    if body is None:
        extracted.warnings.append("DOCX has no word/body element")
        return extracted

    for child in body:
        if child.tag == f"{WORD_NS}p":
            text = _text_from_element(child)
            if text:
                extracted.paragraphs.append(text)
        elif child.tag == f"{WORD_NS}tbl":
            table = _parse_table(child)
            if table:
                extracted.tables.append(table)
    return extracted
