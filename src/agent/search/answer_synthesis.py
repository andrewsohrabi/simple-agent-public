from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from agent.search.schema import SearchHit


class AnswerSynthesizer(Protocol):
    def synthesize(
        self,
        *,
        query: str,
        hits: list[SearchHit],
        citation_rows: list[dict[str, object]],
        model: str,
        max_input_chars: int,
    ) -> str | None:
        ...


class OpenAIAnswerSynthesizer:
    """Small wrapper so tests can replace model-written answer synthesis."""

    def synthesize(
        self,
        *,
        query: str,
        hits: list[SearchHit],
        citation_rows: list[dict[str, object]],
        model: str,
        max_input_chars: int,
    ) -> str | None:
        from openai import OpenAI

        client = OpenAI()
        prompt = build_synthesis_prompt(
            query=query,
            hits=hits,
            citation_rows=citation_rows,
            max_input_chars=max_input_chars,
        )
        response = client.responses.create(model=model, input=prompt)
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return _extract_response_text(response)


def build_synthesis_prompt(
    *,
    query: str,
    hits: list[SearchHit],
    citation_rows: list[dict[str, object]],
    max_input_chars: int,
) -> str:
    source_map = _source_map(citation_rows)
    evidence_blocks: list[str] = []
    for hit in hits:
        label = source_map.get(str(hit.chunk_id))
        if label is None:
            continue
        excerpt = " ".join(hit.text.split())
        evidence_blocks.append(
            "\n".join(
                [
                    (
                        f"[{label}] type={hit.evidence_type}, support={hit.support_level}, "
                        f"doc={hit.doc_id} Rev {hit.revision} - {hit.title}"
                    ),
                    f"Section: {hit.section}",
                    f"Heading path: {' > '.join(hit.heading_path) if hit.heading_path else hit.section}",
                    _table_context(hit),
                    f"Chunk ID: {hit.chunk_id}",
                    f"Evidence: {excerpt}",
                ]
            )
        )

    evidence = "\n\n".join(evidence_blocks)
    if len(evidence) > max_input_chars:
        evidence = evidence[:max_input_chars].rsplit(" ", 1)[0]
        evidence += "\n\n[Evidence truncated to configured answer synthesis limit.]"

    return (
        "You are answering questions about the MedAI QMS internal document corpus.\n"
        "Domain map: BOM means bill of materials; DHF means design history file; "
        "ECR/DCO are change-control records; RSK/RMF/PFMEA/DFMEA are risk evidence; "
        "VVAM is the verification/validation trace matrix; VVPR records are verification "
        "protocol/report evidence; 3P records are third-party reports; MEMO records often "
        "summarize reviews, clearances, and V&V status. Prefer latest active records unless "
        "the question asks for obsolete/history. For traceability, reason in the order "
        "risk/RMF source -> VVAM bridge -> summary/result -> verification target.\n"
        "Use only the evidence below. Every factual claim must cite one or more "
        "provided source labels such as [S1]. Do not invent source labels, document "
        "IDs, revisions, sections, counts, or requirements. If the evidence is "
        "insufficient, say what is missing. Treat metadata evidence as document-discovery "
        "support only; table row facts require table_row or table_full evidence with the "
        "right row/columns.\n\n"
        f"Question:\n{query}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Answer with concise, source-grounded prose."
    )


def _table_context(hit: SearchHit) -> str:
    parts: list[str] = []
    if hit.table_index is not None:
        parts.append(f"table={hit.table_index}")
    if hit.row_start is not None:
        row = f"row={hit.row_start}"
        if hit.row_end is not None and hit.row_end != hit.row_start:
            row += f"-{hit.row_end}"
        parts.append(row)
    if hit.columns:
        parts.append("columns=[" + ", ".join(hit.columns) + "]")
    if hit.row_cells:
        cells = " | ".join(f"{key}={value}" for key, value in hit.row_cells.items())
        parts.append(f"row_cells={cells}")
    return "Table context: " + "; ".join(parts) if parts else "Table context: none"


def uses_only_known_source_labels(
    text: str,
    citation_rows: Sequence[dict[str, object]],
) -> bool:
    known = {f"S{index}" for index, _row in enumerate(citation_rows, start=1)}
    labels = set(re.findall(r"\[(S\d+)\]", text))
    return bool(labels) and labels.issubset(known)


def _source_map(citation_rows: Sequence[dict[str, object]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for index, citation in enumerate(citation_rows, start=1):
        chunk_id = citation.get("chunk_id")
        if chunk_id:
            labels[str(chunk_id)] = f"S{index}"
    return labels


def _extract_response_text(response: object) -> str | None:
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return None
    parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts) if parts else None
