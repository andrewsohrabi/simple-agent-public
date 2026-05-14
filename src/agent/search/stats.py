from __future__ import annotations

from pathlib import Path

from agent.config import SearchConfig
from agent.search.faiss_store import LocalVectorIndex
from agent.search.rerank import configured_reranker_status
from agent.search.sqlite_store import SearchStore


def collect_stats(config: SearchConfig) -> dict[str, object]:
    store = SearchStore(config.index_dir / "qms.sqlite")
    vector_index = LocalVectorIndex(config.index_dir, config)
    vector_stats = vector_index.stats()
    hosted_state = config.openai_vector_store_state
    hosted = {"exists": hosted_state.exists(), "state_path": str(hosted_state)}
    if hosted_state.exists():
        hosted["bytes"] = hosted_state.stat().st_size
        try:
            import json

            state = json.loads(hosted_state.read_text(encoding="utf-8"))
            hosted.update(
                {
                    "status": state.get("status"),
                    "vector_store_id": state.get("vector_store_id"),
                    "corpus_hash": state.get("corpus_hash"),
                    "file_count": state.get("file_count", len(state.get("files", []))),
                    "updated_at": state.get("updated_at"),
                }
            )
        except Exception:
            hosted["status"] = "unreadable"
    return {
        "model_config": config.model_config(),
        "paths": {
            "corpus_zip": str(config.corpus_zip),
            "index_dir": str(config.index_dir),
            "sqlite": str(store.db_path),
            "hosted_state": str(config.openai_vector_store_state),
        },
        "corpus_zip_exists": Path(config.corpus_zip).exists(),
        "sqlite": store.stats(),
        "vector_index": vector_stats,
        "artifact_validation": vector_stats.get("validation", {}),
        "hosted_file_search": hosted,
        "reranker": configured_reranker_status(config),
    }
