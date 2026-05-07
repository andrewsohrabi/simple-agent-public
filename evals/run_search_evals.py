"""CLI harness for validating and scoring QMS search eval datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evals.dataset_schema import dataset_path, load_dataset, summarize_dataset
from evals.reporting import (
    PASS_THRESHOLD,
    aggregate_scores,
    score_case,
    write_json_report,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="smoke",
        help="Built-in dataset name ('smoke' or 'core') or path to JSONL.",
    )
    parser.add_argument(
        "--answers-jsonl",
        help=(
            "Optional JSONL file with rows shaped as "
            "{'id': str, 'answer': str, 'source_ids': list[str]}."
        ),
    )
    parser.add_argument(
        "--report",
        help="Optional directory for timestamped Markdown and JSON reports.",
    )
    parser.add_argument("--json-report", help="Optional path for JSON score output.")
    parser.add_argument(
        "--markdown-report", help="Optional path for Markdown score output."
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=PASS_THRESHOLD,
        help="Average score threshold used when --answers-jsonl is supplied.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of cases to score from the beginning of the dataset.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        help="Search mode to use when running the local search service.",
    )
    parser.add_argument(
        "--hash-embeddings",
        action="store_true",
        help="Use deterministic hash query embeddings for locally built smoke indexes.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate and summarize the dataset; do not run search.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = dataset_path(args.dataset)
    cases = load_dataset(path)
    if args.limit is not None:
        cases = cases[: args.limit]

    if args.validate_only:
        summary = summarize_dataset(cases)
        print(json.dumps({"dataset": str(path), "summary": summary}, indent=2))
        return 0

    if args.answers_jsonl:
        answers = load_answers(Path(args.answers_jsonl))
    else:
        answers = run_search_answers(cases, mode=args.mode, use_hash_embeddings=args.hash_embeddings)

    scores = [
        score_case(
            case,
            answers.get(case.id, {}).get("answer", ""),
            source_ids=answers.get(case.id, {}).get("source_ids", []),
            pass_threshold=args.fail_under,
        )
        for case in cases
    ]

    run_context = build_run_context(mode=args.mode)

    if args.json_report:
        write_json_report(args.json_report, scores, run_context=run_context)
    if args.markdown_report:
        write_markdown_report(
            args.markdown_report,
            scores,
            dataset_name=str(path),
            pass_threshold=args.fail_under,
            run_context=run_context,
        )
    if args.report:
        from datetime import datetime

        report_dir = Path(args.report)
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        run_context["timestamp"] = stamp
        write_json_report(report_dir / f"{stamp}.json", scores, run_context=run_context)
        write_markdown_report(
            report_dir / f"{stamp}.md",
            scores,
            dataset_name=str(path),
            pass_threshold=args.fail_under,
            run_context=run_context,
        )

    summary = aggregate_scores(scores)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["average_score"] >= args.fail_under else 1


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            answer_id = record.get("id")
            if not isinstance(answer_id, str) or not answer_id:
                raise ValueError(f"{path}:{line_number}: answer row missing string id")
            answer = record.get("answer", "")
            if not isinstance(answer, str):
                raise ValueError(f"{path}:{line_number}: answer must be a string")
            source_ids = record.get("source_ids", [])
            if not isinstance(source_ids, list) or not all(
                isinstance(source, str) for source in source_ids
            ):
                raise ValueError(f"{path}:{line_number}: source_ids must be list[str]")
            answers[answer_id] = {"answer": answer, "source_ids": source_ids}
    return answers


def run_search_answers(cases: list[Any], *, mode: str, use_hash_embeddings: bool) -> dict[str, dict[str, Any]]:
    from dotenv import load_dotenv

    from agent.config import load_config
    from agent.search.service import QmsSearchService

    load_dotenv()
    service = QmsSearchService(load_config(), use_hash_embeddings=use_hash_embeddings)
    answers: dict[str, dict[str, Any]] = {}
    for case in cases:
        result = service.search(case.prompt, mode=mode)
        citations = result.get("citations", [])
        source_ids = []
        if isinstance(citations, list):
            for citation in citations:
                if isinstance(citation, dict):
                    source_ids.append(str(citation.get("doc_id", "")))
        answers[case.id] = {
            "answer": str(result.get("answer", "")),
            "source_ids": source_ids,
        }
    return answers


def build_run_context(*, mode: str) -> dict[str, Any]:
    from dotenv import load_dotenv

    from agent.config import load_config
    from agent.search.faiss_store import LocalVectorIndex
    from agent.search.ingest import load_ingest_manifest

    load_dotenv()
    config = load_config()
    index_manifest = LocalVectorIndex(config.index_dir, config).manifest() or {}
    ingest_manifest = load_ingest_manifest(config.index_dir) or {}
    return {
        "timestamp": "unknown",
        "git_commit": _git_commit(),
        "retrieval_mode": mode,
        "corpus_hash": ingest_manifest.get("source_sha256")
        or index_manifest.get("corpus_hash")
        or "unknown",
        "index_manifest": str(config.index_dir / "manifest.json"),
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
        "vector_index": config.vector_index,
        "faiss_index_type": config.faiss_index_type,
        "chat_model": config.chat_model,
        "agent_model": config.agent_model,
        "query_model": config.query_model,
        "reranker_model": config.reranker_model if config.reranker_enabled else "disabled",
        "chunk_count": index_manifest.get("chunk_count", "unknown"),
        "document_count": ingest_manifest.get("document_count", "unknown"),
        "metadata_only_count": ingest_manifest.get("metadata_only_count", "unknown"),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
