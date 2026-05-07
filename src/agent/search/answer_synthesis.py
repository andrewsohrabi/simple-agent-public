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
                    f"[{label}] {hit.doc_id} Rev {hit.revision} - {hit.title}",
                    f"Section: {hit.section}",
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
        "Use only the evidence below. Every factual claim must cite one or more "
        "provided source labels such as [S1]. Do not invent source labels, document "
        "IDs, revisions, sections, counts, or requirements. If the evidence is "
        "insufficient, say what is missing.\n\n"
        f"Question:\n{query}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Answer with concise, source-grounded prose."
    )


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
