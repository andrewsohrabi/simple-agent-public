from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from agent.config import load_config
from agent.search.chunking import chunks_from_manifest
from agent.search.embeddings import HashEmbeddingProvider, OpenAIEmbeddingProvider
from agent.search.faiss_store import LocalVectorIndex
from agent.search.hosted_sync import sync_openai_vector_store
from agent.search.ingest import ingest_corpus, load_ingest_manifest
from agent.search.service import QmsSearchService
from agent.search.sqlite_store import SearchStore
from agent.search.stats import collect_stats


def ingest_main() -> None:
    load_dotenv()
    config = load_config()
    parser = argparse.ArgumentParser(description="Ingest the MedAI QMS DOCX corpus")
    parser.add_argument("--zip", default=str(config.corpus_zip))
    parser.add_argument("--index-dir", default=str(config.index_dir))
    args = parser.parse_args()
    manifest = ingest_corpus(Path(args.zip), Path(args.index_dir))
    store = SearchStore(Path(args.index_dir) / "qms.sqlite")
    chunks = store.load_manifest(manifest, config=config)
    print(
        json.dumps(
            {
                "documents": manifest["document_count"],
                "skipped_empty": manifest["skipped_empty_count"],
                "metadata_only": manifest.get("metadata_only_count", 0),
                "chunks": len(chunks),
                "index_dir": args.index_dir,
            },
            indent=2,
        )
    )


def build_index_main() -> None:
    load_dotenv()
    config = load_config()
    parser = argparse.ArgumentParser(description="Build the local QMS FAISS-compatible index")
    parser.add_argument("--index-dir", default=str(config.index_dir))
    parser.add_argument(
        "--hash-embeddings",
        action="store_true",
        help="Use deterministic hash embeddings for local smoke tests instead of OpenAI.",
    )
    args = parser.parse_args()
    index_dir = Path(args.index_dir)
    manifest = load_ingest_manifest(index_dir)
    if manifest is None:
        raise SystemExit(f"No ingest manifest found in {index_dir}; run ingest-qms first")
    chunks = chunks_from_manifest(manifest, config=config)
    provider = (
        HashEmbeddingProvider(config.embedding_dimensions, model=config.embedding_model)
        if args.hash_embeddings
        else OpenAIEmbeddingProvider(config)
    )
    vector_index = LocalVectorIndex(index_dir, config)
    index_manifest = vector_index.build(
        chunks, provider, corpus_hash=str(manifest.get("source_sha256", ""))
    )
    print(json.dumps(index_manifest, indent=2))


def status_main() -> None:
    load_dotenv()
    config = load_config()
    parser = argparse.ArgumentParser(description="Show MedAI QMS search milestone status")
    parser.add_argument("--tasks", default="TASKS.md")
    parser.add_argument("--index-dir", default=str(config.index_dir))
    parser.add_argument(
        "--openai-state", default=str(config.openai_vector_store_state)
    )
    args = parser.parse_args()
    # Keep the user-requested flags stable while using config for model details.
    from dataclasses import replace

    config = replace(
        config,
        index_dir=Path(args.index_dir),
        openai_vector_store_state=Path(args.openai_state),
    )
    stats = collect_stats(config)
    task_summary = {"exists": Path(args.tasks).exists(), "path": args.tasks}
    if Path(args.tasks).exists():
        text = Path(args.tasks).read_text(encoding="utf-8")
        task_summary["complete_markers"] = text.count("[x]")
        task_summary["pending_markers"] = text.count("[ ]")
    print(json.dumps({"tasks": task_summary, **stats}, indent=2))


def query_main() -> None:
    load_dotenv()
    config = load_config()
    parser = argparse.ArgumentParser(description="Run a MedAI QMS search query")
    parser.add_argument("query")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "local", "hybrid", "hosted"],
    )
    parser.add_argument("--limit", default=16, type=int)
    parser.add_argument(
        "--hash-embeddings",
        action="store_true",
        help="Use deterministic hash embeddings for local smoke tests instead of OpenAI.",
    )
    args = parser.parse_args()
    service = QmsSearchService(
        config,
        use_hash_embeddings=args.hash_embeddings or config.use_hash_embeddings,
    )
    result = service.search(args.query, mode=args.mode, limit=args.limit)
    print(json.dumps(result, indent=2))


def sync_openai_file_search_main() -> None:
    load_dotenv()
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Sync normalized MedAI QMS Markdown files to OpenAI File Search"
    )
    parser.add_argument("--index-dir", default=str(config.index_dir))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create a fresh hosted vector store even when the corpus hash matches.",
    )
    args = parser.parse_args()
    index_dir = Path(args.index_dir)
    manifest = load_ingest_manifest(index_dir)
    if manifest is None:
        raise SystemExit(f"No ingest manifest found in {index_dir}; run ingest-qms first")
    normalized_dir = index_dir / "normalized"
    if not normalized_dir.exists():
        raise SystemExit(f"No normalized Markdown directory found: {normalized_dir}")

    from dataclasses import replace

    config = replace(config, index_dir=index_dir)
    state = sync_openai_vector_store(
        config,
        normalized_dir,
        corpus_hash=str(manifest.get("source_sha256", "")),
        force=args.force,
    )
    print(
        json.dumps(
            {
                "status": state.get("status"),
                "vector_store_id": state.get("vector_store_id"),
                "corpus_hash": state.get("corpus_hash"),
                "file_count": state.get("file_count", len(state.get("files", []))),
                "state_path": str(config.openai_vector_store_state),
            },
            indent=2,
        )
    )
